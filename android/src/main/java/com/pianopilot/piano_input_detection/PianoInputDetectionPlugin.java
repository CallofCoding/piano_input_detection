package com.pianopilot.piano_input_detection;

import org.tensorflow.lite.Interpreter;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.util.Log;

import androidx.annotation.NonNull;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.EventChannel;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;

/**
 * PianoInputDetectionPlugin
 *
 * Detects piano notes from microphone audio using Google Magenta's
 * onsets_frames_wavinput.tflite model.
 *
 * Exposes TWO EventChannels to Flutter:
 *
 * 1. piano_input_detection/events  — clean noteOn / noteOff events
 *    Each emission: List of { "type": "noteOn"|"noteOff", "note": int (MIDI 0-127), "velocity": double }
 *
 * 2. piano_input_detection/raw     — raw tensor output per inference frame
 *    Each emission: List of { "key": int (0-87), "frame": double, "onset": double,
 *                              "offset": double, "velocity": double }
 *
 * MIDI note = Magenta key + 21  (A0=21, Middle C=60, C8=108)
 */
public class PianoInputDetectionPlugin implements FlutterPlugin, MethodCallHandler {

    // ─── Channel names ────────────────────────────────────────────────────────
    private static final String METHOD_CHANNEL = "piano_input_detection";
    private static final String EVENT_CHANNEL  = "piano_input_detection/events"; // noteOn/noteOff
    private static final String RAW_CHANNEL    = "piano_input_detection/raw";    // raw tensors

    // ─── Logging ──────────────────────────────────────────────────────────────
    private static final String LOG_TAG = "PianoInputDetection";

    // ─── Model config ─────────────────────────────────────────────────────────
    private static final String MODEL_FILENAME   = "onsets_frames_wavinput.tflite";
    private static final int    SAMPLE_RATE      = 16000;
    private static final int    RECORDING_LENGTH = 17920;
    private static final int    OUT_STEP_NOTES   = 32;
    private static final int    PIANO_KEYS       = 88;

    // ─── Detection thresholds ─────────────────────────────────────────────────
    // Raw logits. 0.0f = sigmoid(0) = 0.5 probability.
    private static final float ONSET_THRESHOLD  = 1.0f;
    private static final float FRAME_THRESHOLD  = 0.0f;
    private static final float OFFSET_THRESHOLD = 0.0f;

    // MIDI offset: Magenta key 0 (A0) = MIDI note 21
    private static final int MIDI_OFFSET = 21;

    // ─── Flutter channels ─────────────────────────────────────────────────────
    private MethodChannel          methodChannel;
    private EventChannel           noteEventChannel;
    private EventChannel           rawEventChannel;
    private EventChannel.EventSink noteEventSink;
    private EventChannel.EventSink rawSink;
    private Result                 pendingResult;

    // ─── Android ──────────────────────────────────────────────────────────────
    private Context context;
    private final Handler handler = new Handler(Looper.getMainLooper());

    // ─── Audio ────────────────────────────────────────────────────────────────
    private AudioRecord      audioRecord;
    private Thread           recordingThread;
    private Thread           recognitionThread;
    private volatile boolean shouldContinue            = false;
    private volatile boolean shouldContinueRecognition = false;

    private final short[]       recordingBuffer     = new short[RECORDING_LENGTH];
    private int                 recordingOffset     = 0;
    private final ReentrantLock recordingBufferLock = new ReentrantLock();

    // ─── TFLite ───────────────────────────────────────────────────────────────
    private Interpreter tfLite = null;

    // ─── Note state (indexed by Magenta key 0–87) ─────────────────────────────
    private final boolean[] activeNotes = new boolean[PIANO_KEYS];

    // ─── Onset cooldown ───────────────────────────────────────────────────────
    // After a noteOn fires for a key, any further onsets for that same key
    // within ONSET_COOLDOWN_MS are ignored. This prevents the model emitting
    // multiple onset detections across consecutive frames for a single physical
    // key press from reaching Flutter as duplicate noteOn events.
    private static final long ONSET_COOLDOWN_MS = 100;
    private final long[] lastNoteOnTime = new long[PIANO_KEYS];

    // ─────────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding binding) {
        context = binding.getApplicationContext();

        methodChannel = new MethodChannel(binding.getBinaryMessenger(), METHOD_CHANNEL);
        methodChannel.setMethodCallHandler(this);

        // Channel 1: noteOn / noteOff events
        noteEventChannel = new EventChannel(binding.getBinaryMessenger(), EVENT_CHANNEL);
        noteEventChannel.setStreamHandler(new EventChannel.StreamHandler() {
            @Override
            public void onListen(Object args, EventChannel.EventSink sink) {
                noteEventSink = sink;
            }
            @Override
            public void onCancel(Object args) {
                noteEventSink = null;
            }
        });

        // Channel 2: raw tensor output
        rawEventChannel = new EventChannel(binding.getBinaryMessenger(), RAW_CHANNEL);
        rawEventChannel.setStreamHandler(new EventChannel.StreamHandler() {
            @Override
            public void onListen(Object args, EventChannel.EventSink sink) {
                rawSink = sink;
            }
            @Override
            public void onCancel(Object args) {
                rawSink = null;
            }
        });
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        methodChannel.setMethodCallHandler(null);
        noteEventChannel.setStreamHandler(null);
        rawEventChannel.setStreamHandler(null);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MethodChannel
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
        this.pendingResult = result;
        switch (call.method) {
            case "prepare":
                loadModel();
                break;
            case "start":
                startRecord();
                startRecognition();
                result.success(true);
                break;
            case "stop":
                stopRecording();
                stopRecognition();
                result.success(true);
                break;
            default:
                result.notImplemented();
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Model loading
    // ─────────────────────────────────────────────────────────────────────────

    private void loadModel() {
        Log.v(LOG_TAG, "Loading TFLite model...");
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            tfLite = new Interpreter(loadModelFile(context, MODEL_FILENAME), options);
            tfLite.resizeInput(0, new int[]{RECORDING_LENGTH, 1});
            Log.v(LOG_TAG, "TFLite model loaded successfully");
            pendingResult.success(true);
        } catch (Exception e) {
            Log.e(LOG_TAG, "Failed to load TFLite model: " + e.getMessage());
            pendingResult.success(false);
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fd.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Recording thread
    // ─────────────────────────────────────────────────────────────────────────

    private synchronized void startRecord() {
        if (recordingThread != null) return;
        shouldContinue = true;
        recordingThread = new Thread(this::record, "PianoInput-Recording");
        recordingThread.start();
    }

    private void record() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) bufferSize = SAMPLE_RATE * 2;

        short[] audioBuffer = new short[bufferSize / 2];
        audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize
        );

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "AudioRecord failed to initialize");
            return;
        }

        audioRecord.startRecording();
        Log.v(LOG_TAG, "Recording started");

        while (shouldContinue) {
            int numberRead = audioRecord.read(audioBuffer, 0, audioBuffer.length);
            if (numberRead <= 0) continue;

            int maxLength          = recordingBuffer.length;
            int newRecordingOffset = recordingOffset + numberRead;
            int secondCopyLength   = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength    = numberRead - secondCopyLength;

            recordingBufferLock.lock();
            try {
                System.arraycopy(audioBuffer, 0,               recordingBuffer, recordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0,               secondCopyLength);
                recordingOffset = newRecordingOffset % maxLength;
            } finally {
                recordingBufferLock.unlock();
            }
        }
    }

    public synchronized void stopRecording() {
        if (recordingThread == null) return;
        shouldContinue = false;
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }
        recordingOffset = 0;
        recordingThread = null;
        Log.d(LOG_TAG, "Recording stopped");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Recognition thread
    // ─────────────────────────────────────────────────────────────────────────

    private synchronized void startRecognition() {
        if (recognitionThread != null) return;
        shouldContinueRecognition = true;
        Arrays.fill(activeNotes, false);
        Arrays.fill(lastNoteOnTime, 0L);
        recognitionThread = new Thread(this::recognize, "PianoInput-Recognition");
        recognitionThread.start();
    }

    private void recognize() {
        Log.v(LOG_TAG, "Recognition started");

        short[]   inputBuffer      = new short[RECORDING_LENGTH];
        float[][] floatInputBuffer  = new float[RECORDING_LENGTH][1];

        float[][][] frameBuffer    = new float[1][OUT_STEP_NOTES][PIANO_KEYS];
        float[][][] onsetBuffer    = new float[1][OUT_STEP_NOTES][PIANO_KEYS];
        float[][][] offsetBuffer   = new float[1][OUT_STEP_NOTES][PIANO_KEYS];
        float[][][] velocityBuffer = new float[1][OUT_STEP_NOTES][PIANO_KEYS];

        while (shouldContinueRecognition) {

            // 1. Copy audio buffer
            recordingBufferLock.lock();
            try {
                int maxLength    = recordingBuffer.length;
                int firstCopyLen = maxLength - recordingOffset;
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0,            firstCopyLen);
                System.arraycopy(recordingBuffer, 0,               inputBuffer, firstCopyLen, recordingOffset);
            } finally {
                recordingBufferLock.unlock();
            }

            // 2. Normalise to [-1, 1]
            for (int i = 0; i < RECORDING_LENGTH; i++) {
                floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f;
            }

            // 3. Run inference
            int idxFrame    = tfLite.getOutputIndex("frame_logits");
            int idxOnset    = tfLite.getOutputIndex("onset_logits");
            int idxOffset   = tfLite.getOutputIndex("offset_logits");
            int idxVelocity = tfLite.getOutputIndex("velocity_values");

            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(idxFrame,    frameBuffer);
            outputMap.put(idxOnset,    onsetBuffer);
            outputMap.put(idxOffset,   offsetBuffer);
            outputMap.put(idxVelocity, velocityBuffer);

            tfLite.runForMultipleInputsOutputs(new Object[]{floatInputBuffer}, outputMap);

            float[][] frames     = ((float[][][]) outputMap.get(idxFrame))[0];
            float[][] onsets     = ((float[][][]) outputMap.get(idxOnset))[0];
            float[][] offsets    = ((float[][][]) outputMap.get(idxOffset))[0];
            float[][] velocities = ((float[][][]) outputMap.get(idxVelocity))[0];

            // 4. Build both event lists in one pass
            List<Map<String, Object>> noteEvents = new ArrayList<>();
            List<Map<String, Object>> rawEvents  = new ArrayList<>();

            for (int frame = 0; frame < frames.length; frame++) {
                for (int key = 0; key < PIANO_KEYS; key++) {

                    float onsetLogit  = onsets[frame][key];
                    float frameLogit  = frames[frame][key];
                    float offsetLogit = offsets[frame][key];
                    float velocity    = velocities[frame][key];

                    boolean isOnset  = onsetLogit  > ONSET_THRESHOLD;
                    boolean isFrame  = frameLogit  > FRAME_THRESHOLD;
                    boolean isOffset = offsetLogit > OFFSET_THRESHOLD;
                    int     midiNote = key + MIDI_OFFSET;

                    // ── noteOn / noteOff ──────────────────────────────────
                    long now = System.currentTimeMillis();
                    boolean inCooldown = (now - lastNoteOnTime[key]) < ONSET_COOLDOWN_MS;

                    if (isOnset && !activeNotes[key] && !inCooldown) {
                        // New noteOn — key was not active and cooldown has passed
                        activeNotes[key] = true;
                        lastNoteOnTime[key] = now;
                        Map<String, Object> e = new HashMap<>();
                        e.put("type",     "noteOn");
                        e.put("note",     midiNote);
                        e.put("velocity", (double) Math.max(0.0f, Math.min(1.0f, velocity)));
                        noteEvents.add(e);

                    } else if (activeNotes[key] && (isOffset || !isFrame)) {
                        // noteOff — frame dropped or offset fired
                        activeNotes[key] = false;
                        Map<String, Object> e = new HashMap<>();
                        e.put("type",     "noteOff");
                        e.put("note",     midiNote);
                        e.put("velocity", 0.0);
                        noteEvents.add(e);
                    }

                    // ── raw (only entries with activity) ─────────────────
                    if (frameLogit > 0 || onsetLogit > 0) {
                        Map<String, Object> r = new HashMap<>();
                        r.put("key",      key);
                        r.put("frame",    (double) frameLogit);
                        r.put("onset",    (double) onsetLogit);
                        r.put("offset",   (double) offsetLogit);
                        r.put("velocity", (double) velocity);
                        rawEvents.add(r);
                    }
                }
            }

            // 4.5 Small sleep to avoid busy-looping and reduce CPU usage
            try { Thread.sleep(20); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }

            // 5. Emit to both channels independently
            if (!noteEvents.isEmpty()) emitToSink(noteEventSink, noteEvents);
            if (!rawEvents.isEmpty())  emitToSink(rawSink, rawEvents);
        }

        // Cleanup: fire noteOff for any still-active notes on stop
        List<Map<String, Object>> cleanupEvents = new ArrayList<>();
        for (int key = 0; key < PIANO_KEYS; key++) {
            if (activeNotes[key]) {
                activeNotes[key] = false;
                Map<String, Object> e = new HashMap<>();
                e.put("type",     "noteOff");
                e.put("note",     key + MIDI_OFFSET);
                e.put("velocity", 0.0);
                cleanupEvents.add(e);
            }
        }
        if (!cleanupEvents.isEmpty()) emitToSink(noteEventSink, cleanupEvents);

        Log.d(LOG_TAG, "Recognition stopped");
    }

    public synchronized void stopRecognition() {
        if (recognitionThread == null) return;
        shouldContinueRecognition = false;
        recognitionThread = null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Emit helpers
    // ─────────────────────────────────────────────────────────────────────────

    private void emitToSink(final EventChannel.EventSink sink, final List<Map<String, Object>> data) {
        runOnUIThread(() -> { if (sink != null) sink.success(data); });
    }

    private void runOnUIThread(Runnable runnable) {
        if (Looper.getMainLooper() == Looper.myLooper()) runnable.run();
        else handler.post(runnable);
    }
}