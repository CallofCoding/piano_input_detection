import 'dart:async';
import 'package:flutter/services.dart';
import 'src/note_event.dart';

export 'src/note_event.dart';

/// State of the TFLite model.
enum PianoDetectionState { idle, loading, ready, error }

/// **PianoInputDetection**
///
/// A Flutter plugin that detects piano notes in real-time from the device
/// microphone using Google Magenta's `onsets_frames_wavinput.tflite` model.
///
/// Emits clean [NoteEvent] objects with [NoteEventType.noteOn] /
/// [NoteEventType.noteOff] and a MIDI note number (0–127).
///
/// ### Setup
/// 1. Download `onsets_frames_wavinput.tflite` from Google Magenta and place
///    it in `android/app/src/main/assets/`.
/// 2. Add `RECORD_AUDIO` permission to your `AndroidManifest.xml`.
/// 3. Add `aaptOptions { noCompress 'tflite' }` to your app `build.gradle`.
///
/// ### Usage
/// ```dart
/// final detector = PianoInputDetection();
///
/// await detector.prepare();   // load TFLite model
/// await detector.start();     // begin mic capture + inference
///
/// detector.noteEvents.listen((NoteEvent event) {
///   print('${event.type} — MIDI note: ${event.note}');
/// });
///
/// await detector.stop();      // stop mic + inference
/// ```
class PianoInputDetection {
  static const MethodChannel _methodChannel =
      MethodChannel('piano_input_detection');

  static const EventChannel _eventChannel =
      EventChannel('piano_input_detection/events');

  PianoDetectionState _state = PianoDetectionState.idle;

  /// Current state of the TFLite model / engine.
  PianoDetectionState get state => _state;

  // Internal broadcast stream controller — keeps a single native subscription
  // but allows multiple Dart listeners.
  StreamController<NoteEvent>? _controller;
  StreamSubscription<dynamic>? _nativeSubscription;

  /// Stream of [NoteEvent]s emitted in real-time.
  ///
  /// Each event carries:
  /// - [NoteEvent.type]     — [NoteEventType.noteOn] or [NoteEventType.noteOff]
  /// - [NoteEvent.note]     — MIDI note number (0–127)
  /// - [NoteEvent.velocity] — 0.0–1.0 (0.0 for noteOff)
  ///
  /// The stream is a broadcast stream — safe to listen to from multiple places.
  Stream<NoteEvent> get noteEvents {
    _ensureController();
    return _controller!.stream;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /// Loads the TFLite model. Must be called once before [start].
  ///
  /// Returns `true` if the model loaded successfully, `false` otherwise.
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
  /// Call [prepare] first. Listen to [noteEvents] to receive [NoteEvent]s.
  Future<void> start() async {
    _ensureController();
    _subscribeToNative();
    try {
      await _methodChannel.invokeMethod<void>('start');
    } catch (e) {
      _log('start() failed: $e');
    }
  }

  /// Stops microphone capture and inference.
  ///
  /// Any currently active notes will receive a [NoteEventType.noteOff] event
  /// automatically before the stream goes quiet.
  Future<void> stop() async {
    try {
      await _methodChannel.invokeMethod<void>('stop');
    } catch (e) {
      _log('stop() failed: $e');
    }
    await _unsubscribeFromNative();
  }

  /// Disposes all resources. Call when the plugin is no longer needed.
  Future<void> dispose() async {
    await stop();
    await _controller?.close();
    _controller = null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal helpers
  // ─────────────────────────────────────────────────────────────────────────

  void _ensureController() {
    if (_controller == null || _controller!.isClosed) {
      _controller = StreamController<NoteEvent>.broadcast();
    }
  }

  void _subscribeToNative() {
    if (_nativeSubscription != null) return;

    _nativeSubscription = _eventChannel
        .receiveBroadcastStream()
        .listen(
          _onNativeEvent,
          onError: _onNativeError,
          cancelOnError: false,
        );
  }

  Future<void> _unsubscribeFromNative() async {
    await _nativeSubscription?.cancel();
    _nativeSubscription = null;
  }

  /// The native layer emits a List of event maps per inference frame.
  /// We unpack each map into a [NoteEvent] and push it to the controller.
  void _onNativeEvent(dynamic rawList) {
    if (_controller == null || _controller!.isClosed) return;
    if (rawList is! List) return;

    for (final raw in rawList) {
      if (raw is Map) {
        try {
          final event = NoteEvent.fromMap(raw.cast<Object?, Object?>());
          _controller!.add(event);
        } catch (e) {
          _log('Failed to parse NoteEvent: $e  raw=$raw');
        }
      }
    }
  }

  void _onNativeError(dynamic error) {
    _log('Native stream error: $error');
    if (_controller != null && !_controller!.isClosed) {
      _controller!.addError(error);
    }
  }

  void _log(String msg) {
    // ignore: avoid_print
    print('[PianoInputDetection] $msg');
  }
}
