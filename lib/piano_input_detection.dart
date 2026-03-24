import 'dart:async';
import 'package:flutter/services.dart';
import 'src/note_event.dart';
import 'src/raw_note_data.dart';

export 'src/note_event.dart';
export 'src/raw_note_data.dart';

/// State of the TFLite model.
enum PianoDetectionState { idle, loading, ready, error }

/// **PianoInputDetection**
///
/// Detects piano notes in real-time from the device microphone using
/// Google Magenta's `onsets_frames_wavinput.tflite` model.
///
/// Exposes two streams:
///
/// - [noteEvents]           — clean [NoteEvent] stream (noteOn / noteOff)
///                            with MIDI note numbers (0–127)
/// - [startAudioRecognition] — raw tensor stream, same format as the
///                             original flutter_piano_audio_detection library
///
/// ### Usage
/// ```dart
/// final detector = PianoInputDetection();
/// await detector.prepare();
/// await detector.start();
///
/// // Clean stream
/// detector.noteEvents.listen((event) {
///   print('${event.type} note: ${event.note}');
/// });
///
/// // Raw stream
/// detector.startAudioRecognition().listen((frames) {
///   for (final f in frames) {
///     print('key=${f.key} onset=${f.onset}');
///   }
/// });
///
/// await detector.stop();
/// await detector.dispose();
/// ```
class PianoInputDetection {
  static const MethodChannel _methodChannel =
  MethodChannel('piano_input_detection');

  static const EventChannel _noteEventChannel =
  EventChannel('piano_input_detection/events');

  static const EventChannel _rawChannel =
  EventChannel('piano_input_detection/raw');

  PianoDetectionState _state = PianoDetectionState.idle;

  /// Current state of the TFLite model.
  PianoDetectionState get state => _state;

  // ── Internal broadcast controllers ────────────────────────────────────────
  StreamController<NoteEvent>?   _noteController;
  StreamController<List<RawNoteData>>? _rawController;

  StreamSubscription<dynamic>? _nativeNoteSubscription;
  StreamSubscription<dynamic>? _nativeRawSubscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /// Stream of [NoteEvent]s — noteOn / noteOff with MIDI note numbers (0–127).
  ///
  /// Broadcast stream, safe to listen to from multiple places.
  Stream<NoteEvent> get noteEvents {
    _ensureNoteController();
    return _noteController!.stream;
  }

  /// Raw tensor stream — same format as the original library.
  ///
  /// Each emission is a [List<RawNoteData>] containing all active keys
  /// for that inference frame with their raw logit values.
  ///
  /// Useful for debugging, visualisation, or building your own logic
  /// on top of the raw model output.
  Stream<List<RawNoteData>> startAudioRecognition() {
    _ensureRawController();
    return _rawController!.stream;
  }

  /// Loads the TFLite model. Must be called once before [start].
  ///
  /// Returns `true` if successful.
  Future<bool> prepare() async {
    _state = PianoDetectionState.loading;
    try {
      final bool success =
          await _methodChannel.invokeMethod<bool>('prepare') ?? false;
      _state = success ? PianoDetectionState.ready : PianoDetectionState.error;
      return success;
    } catch (e) {
      _state = PianoDetectionState.error;
      _log('prepare() failed: $e');
      return false;
    }
  }

  /// Starts microphone capture and real-time inference.
  ///
  /// Call [prepare] first. Subscribe to [noteEvents] and/or
  /// [startAudioRecognition] to receive data.
  Future<void> start() async {
    _ensureNoteController();
    _ensureRawController();
    _subscribeToNoteChannel();
    _subscribeToRawChannel();
    try {
      await _methodChannel.invokeMethod<void>('start');
    } catch (e) {
      _log('start() failed: $e');
    }
  }

  /// Stops microphone capture and inference.
  ///
  /// Any held notes will automatically receive a [NoteEventType.noteOff]
  /// before the stream goes quiet.
  Future<void> stop() async {
    try {
      await _methodChannel.invokeMethod<void>('stop');
    } catch (e) {
      _log('stop() failed: $e');
    }
    await _unsubscribeAll();
  }

  /// Disposes all resources. Call when the plugin is no longer needed.
  Future<void> dispose() async {
    await stop();
    await _noteController?.close();
    await _rawController?.close();
    _noteController = null;
    _rawController  = null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal helpers
  // ─────────────────────────────────────────────────────────────────────────

  void _ensureNoteController() {
    if (_noteController == null || _noteController!.isClosed) {
      _noteController = StreamController<NoteEvent>.broadcast();
    }
  }

  void _ensureRawController() {
    if (_rawController == null || _rawController!.isClosed) {
      _rawController = StreamController<List<RawNoteData>>.broadcast();
    }
  }

  void _subscribeToNoteChannel() {
    if (_nativeNoteSubscription != null) return;
    _nativeNoteSubscription = _noteEventChannel
        .receiveBroadcastStream()
        .listen(_onNoteEvent, onError: _onNoteError, cancelOnError: false);
  }

  void _subscribeToRawChannel() {
    if (_nativeRawSubscription != null) return;
    _nativeRawSubscription = _rawChannel
        .receiveBroadcastStream()
        .listen(_onRawEvent, onError: _onRawError, cancelOnError: false);
  }

  Future<void> _unsubscribeAll() async {
    await _nativeNoteSubscription?.cancel();
    await _nativeRawSubscription?.cancel();
    _nativeNoteSubscription = null;
    _nativeRawSubscription  = null;
  }

  void _onNoteEvent(dynamic rawList) {
    if (_noteController == null || _noteController!.isClosed) return;
    if (rawList is! List) return;
    for (final raw in rawList) {
      if (raw is Map) {
        try {
          _noteController!.add(NoteEvent.fromMap(raw.cast<Object?, Object?>()));
        } catch (e) {
          _log('Failed to parse NoteEvent: $e  raw=$raw');
        }
      }
    }
  }

  void _onRawEvent(dynamic rawList) {
    if (_rawController == null || _rawController!.isClosed) return;
    if (rawList is! List) return;
    try {
      final items = rawList
          .whereType<Map>()
          .map((m) => RawNoteData.fromMap(m.cast<Object?, Object?>()))
          .toList();
      if (items.isNotEmpty) _rawController!.add(items);
    } catch (e) {
      _log('Failed to parse RawNoteData: $e');
    }
  }

  void _onNoteError(dynamic error) {
    _log('Note channel error: $error');
    _noteController?.addError(error);
  }

  void _onRawError(dynamic error) {
    _log('Raw channel error: $error');
    _rawController?.addError(error);
  }

  void _log(String msg) {
    // ignore: avoid_print
    print('[PianoInputDetection] $msg');
  }
}
