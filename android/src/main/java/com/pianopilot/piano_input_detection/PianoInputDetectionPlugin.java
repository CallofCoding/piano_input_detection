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
 * Emits proper noteOn / noteOff events over an EventChannel.
 * Note numbers follow the MIDI standard (0–127), where:
 *   - Magenta key index (0–87) maps to MIDI via: midiNote = magentaKey + 21
 *   - A0 = MIDI 21, Middle C = MIDI 60, C8 = MIDI 108
 *
 * Event format emitted to Flutter:
 * {
 *   "type":     "noteOn" | "noteOff",
 *   "note":     int (MIDI 0–127),
 *   "velocity": double (0.0–1.0)   // always 0.0 for noteOff
 * }
 */
public class PianoInputDetectionPlugin implements FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler {

    // ─── Channel names ────────────────────────────────────────────────────────
    private static final String METHOD_CHANNEL  = "piano_input_detection";
    private static final String EVENT_CHANNEL   = "piano_input_detection/events";

    // ─── Logging ──────────────────────────────────────────────────────────────
    private static final String LOG_TAG = "PianoInputDetection";

    // ─── Model config ─────────────────────────────────────────────────────────
    private static final String MODEL_FILENAME  = "onsets_frames_wavinput.tflite";
    private static final int    SAMPLE_RATE     = 16000;   // Google Magenta default
    private static final int    RECORDING_LENGTH = 17920;
    private static final int    OUT_STEP_NOTES  = 32;
    private static final int    PIANO_KEYS      = 88;

    // ─── Detection thresholds (tuned for Magenta logits) ─────────────────────
    // Logits are raw (not yet sigmoid-activated). Threshold 0.0 corresponds to
    // sigmoid(0) = 0.5 probability, which is a good default starting point.
    private static final float ONSET_THRESHOLD  = 0.0f;
    private static final float FRAME_THRESHOLD  = 0.0f;
    private static final float OFFSET_THRESHOLD = 0.0f;

    // MIDI offset: Magenta key 0 = A0 = MIDI note 21
    private static final int MIDI_OFFSET = 21;

    // ─── Flutter channels ─────────────────────────────────────────────────────
    private MethodChannel  methodChannel;
    private EventChannel   eventChannel;
    private EventChannel.EventSink events;
    private Result         pendingResult;

    // ─── Android context ──────────────────────────────────────────────────────
    private Context context;
    private final Handler handler = new Handler(Looper.getMainLooper());

    // ─── Audio recording ──────────────────────────────────────────────────────
    private AudioRecord    audioRecord    = null;
    private Thread         recordingThread;
    private Thread         recognitionThread;
    private volatile boolean shouldContinue          = false;
    private volatile boolean shouldContinueRecognition = false;

    private final short[]         recordingBuffer = new short[RECORDING_LENGTH];
    private int                   recordingOffset = 0;
    private final ReentrantLock   recordingBufferLock = new ReentrantLock();

    // ─── TFLite ───────────────────────────────────────────────────────────────
    private Interpreter tfLite = null;

    // ─── Note state tracking (true = note is currently held/active) ───────────
    // Indexed by Magenta key (0–87). Converted to MIDI when emitting.
    private final boolean[] activeNotes = new boolean[PIANO_KEYS];

    // ─────────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public void onAttachedToEngine(@NonNull FlutterPlugin.FlutterPluginBinding binding) {
        context = binding.getApplicationContext();

        methodChannel = new MethodChannel(binding.getBinaryMessenger(), METHOD_CHANNEL);
        methodChannel.setMethodCallHandler(this);

        eventChannel = new EventChannel(binding.getBinaryMessenger(), EVENT_CHANNEL);
        eventChannel.setStreamHandler(this);
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPlugin.FlutterPluginBinding binding) {
        methodChannel.setMethodCallHandler(null);
        eventChannel.setStreamHandler(null);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EventChannel.StreamHandler
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public void onListen(Object arguments, EventChannel.EventSink eventSink) {
        this.events = eventSink;
    }

    @Override
    public void onCancel(Object arguments) {
        this.events = null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MethodChannel.MethodCallHandler
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
                break;
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
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fileDescriptor.getStartOffset(),
                fileDescriptor.getDeclaredLength()
        );
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

        int bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }

        short[] audioBuffer = new short[bufferSize / 2];

        audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.DEFAULT,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
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

            int maxLength         = recordingBuffer.length;
            int newRecordingOffset = recordingOffset + numberRead;
            int secondCopyLength  = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength   = numberRead - secondCopyLength;

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
        // Reset note state so no ghost noteOffs fire after a restart
        java.util.Arrays.fill(activeNotes, false);
        recognitionThread = new Thread(this::recognize, "PianoInput-Recognition");
        recognitionThread.start();
    }

    private void recognize() {
        Log.v(LOG_TAG, "Recognition started");

        short[]   inputBuffer      = new short[RECORDING_LENGTH];
        float[][] floatInputBuffer  = new float[RECORDING_LENGTH][1];

        // Output tensor buffers — shape [1][OUT_STEP_NOTES][88]
        float[][][] frameBuffer    = new float[1][OUT_STEP_NOTES][PIANO_KEYS];
        float[][][] onsetBuffer    = new float[1][OUT_STEP_NOTES][PIANO_KEYS];
        float[][][] offsetBuffer   = new float[1][OUT_STEP_NOTES][PIANO_KEYS];
        float[][][] velocityBuffer = new float[1][OUT_STEP_NOTES][PIANO_KEYS];

        while (shouldContinueRecognition) {

            // ── 1. Copy latest audio into inputBuffer ──────────────────────
            recordingBufferLock.lock();
            try {
                int maxLength      = recordingBuffer.length;
                int firstCopyLen   = maxLength - recordingOffset;
                int secondCopyLen  = recordingOffset;
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0,            firstCopyLen);
                System.arraycopy(recordingBuffer, 0,               inputBuffer, firstCopyLen, secondCopyLen);
            } finally {
                recordingBufferLock.unlock();
            }

            // ── 2. Normalise to [-1, 1] ────────────────────────────────────
            for (int i = 0; i < RECORDING_LENGTH; i++) {
                floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f;
            }

            // ── 3. Run inference ───────────────────────────────────────────
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

            // ── 4. Derive noteOn / noteOff per frame, per key ──────────────
            List<Map<String, Object>> eventsToEmit = new ArrayList<>();

            for (int frame = 0; frame < frames.length; frame++) {
                for (int key = 0; key < PIANO_KEYS; key++) {

                    float onsetLogit  = onsets[frame][key];
                    float frameLogit  = frames[frame][key];
                    float offsetLogit = offsets[frame][key];
                    float velocity    = velocities[frame][key]; // already 0–1 from model

                    boolean isOnset  = onsetLogit  > ONSET_THRESHOLD;
                    boolean isFrame  = frameLogit  > FRAME_THRESHOLD;
                    boolean isOffset = offsetLogit > OFFSET_THRESHOLD;

                    int midiNote = key + MIDI_OFFSET; // convert to MIDI (21–108)

                    if (isOnset && !activeNotes[key]) {
                        // ── noteOn: onset detected and note was not already active
                        activeNotes[key] = true;
                        Map<String, Object> event = new HashMap<>();
                        event.put("type",     "noteOn");
                        event.put("note",     midiNote);
                        event.put("velocity", (double) Math.max(0.0f, Math.min(1.0f, velocity)));
                        eventsToEmit.add(event);

                    } else if (activeNotes[key] && (isOffset || !isFrame)) {
                        // ── noteOff: note was active but frame dropped or offset fired
                        activeNotes[key] = false;
                        Map<String, Object> event = new HashMap<>();
                        event.put("type",     "noteOff");
                        event.put("note",     midiNote);
                        event.put("velocity", 0.0);
                        eventsToEmit.add(event);
                    }
                }
            }

            // ── 5. Emit collected events to Flutter ────────────────────────
            if (!eventsToEmit.isEmpty()) {
                emitEvents(eventsToEmit);
            }
        }

        // ── When recognition stops: fire noteOff for any still-active notes ──
        List<Map<String, Object>> cleanupEvents = new ArrayList<>();
        for (int key = 0; key < PIANO_KEYS; key++) {
            if (activeNotes[key]) {
                activeNotes[key] = false;
                Map<String, Object> event = new HashMap<>();
                event.put("type",     "noteOff");
                event.put("note",     key + MIDI_OFFSET);
                event.put("velocity", 0.0);
                cleanupEvents.add(event);
            }
        }
        if (!cleanupEvents.isEmpty()) {
            emitEvents(cleanupEvents);
        }

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

    private void emitEvents(List<Map<String, Object>> noteEvents) {
        runOnUIThread(() -> {
            if (events != null) {
                events.success(noteEvents);
            }
        });
    }

    private void runOnUIThread(Runnable runnable) {
        if (Looper.getMainLooper() == Looper.myLooper()) {
            runnable.run();
        } else {
            handler.post(runnable);
        }
    }
}
