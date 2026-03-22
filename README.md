# piano_input_detection

Real-time piano note detection from microphone audio using Google Magenta's
`onsets_frames_wavinput.tflite` TFLite model.

Emits proper **noteOn / noteOff** events with **MIDI-standard note numbers** (0–127).

---

## How it works

The Google Magenta model outputs four tensors per inference frame:
- `onset_logits`  — probability a note **started**
- `frame_logits`  — probability a note is **currently active**
- `offset_logits` — probability a note **ended**
- `velocity_values` — how hard the note was struck

This plugin compares consecutive frames against an internal `activeNotes[88]`
state array and fires discrete events:
- **noteOn**  — when onset fires for a key that wasn't already active
- **noteOff** — when offset fires OR frame drops for a key that was active

---

## MIDI note numbers

| Key         | MIDI note |
|-------------|-----------|
| A0 (lowest) | 21        |
| Middle C    | 60        |
| C8 (highest)| 108       |

Internally the Magenta model uses 0–87. This plugin adds an offset of **+21**
so the output always matches MIDI.

---

## Android setup

### 1. Download the TFLite model

Download `onsets_frames_wavinput.tflite` from Google Magenta:
```
https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite
```
Place it in:
```
android/app/src/main/assets/onsets_frames_wavinput.tflite
```

### 2. AndroidManifest.xml

Add the microphone permission:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

### 3. app/build.gradle

Prevent Gradle from compressing the model file:
```groovy
android {
    aaptOptions {
        noCompress 'tflite'
    }
}
```

### 4. Request runtime permission

Use `permission_handler` or your preferred package to request
`Permission.microphone` before calling `start()`.

---

## Usage

```dart
import 'package:piano_input_detection/piano_input_detection.dart';

final detector = PianoInputDetection();

// 1. Load the TFLite model (once)
final ready = await detector.prepare();
if (!ready) return;

// 2. Start mic capture and inference
await detector.start();

// 3. Listen for events
detector.noteEvents.listen((NoteEvent event) {
  if (event.type == NoteEventType.noteOn) {
    print('NOTE ON  — MIDI: ${event.note}  velocity: ${event.velocity}');
  } else {
    print('NOTE OFF — MIDI: ${event.note}');
  }
});

// 4. Stop when done
await detector.stop();

// 5. Dispose when the widget is destroyed
await detector.dispose();
```

---

## NoteEvent

| Property   | Type            | Description                              |
|------------|-----------------|------------------------------------------|
| `type`     | `NoteEventType` | `noteOn` or `noteOff`                    |
| `note`     | `int`           | MIDI note number (0–127)                 |
| `velocity` | `double`        | 0.0–1.0. Always 0.0 for `noteOff`        |

---

## Detection thresholds

The model outputs raw logits. The plugin uses a threshold of `0.0` by default
(equivalent to sigmoid probability ≥ 0.5). These are defined as constants in
`PianoInputDetectionPlugin.java` and can be tuned if needed:

```java
private static final float ONSET_THRESHOLD  = 0.0f;
private static final float FRAME_THRESHOLD  = 0.0f;
private static final float OFFSET_THRESHOLD = 0.0f;
```
