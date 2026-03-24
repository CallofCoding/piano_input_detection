/// Raw tensor output for a single key from one inference frame.
///
/// This matches the original `flutter_piano_audio_detection` data format.
/// Useful for debugging, visualisation, or building custom detection logic.
///
/// Key is in **Magenta format** (0–87):
/// - 0  = A0 (lowest piano key)
/// - 39 = Middle C (C4)
/// - 87 = C8 (highest piano key)
///
/// To convert to MIDI: `midiNote = key + 21`
class RawNoteData {
  /// Magenta key index (0–87).
  final int key;

  /// Raw frame logit — how active/sustained this note is.
  final double frame;

  /// Raw onset logit — probability this note just started.
  final double onset;

  /// Raw offset logit — probability this note just ended.
  final double offset;

  /// Velocity value (0.0–1.0).
  final double velocity;

  const RawNoteData({
    required this.key,
    required this.frame,
    required this.onset,
    required this.offset,
    required this.velocity,
  });

  factory RawNoteData.fromMap(Map<Object?, Object?> map) {
    return RawNoteData(
      key:      (map['key']      as num).toInt(),
      frame:    (map['frame']    as num).toDouble(),
      onset:    (map['onset']    as num).toDouble(),
      offset:   (map['offset']   as num).toDouble(),
      velocity: (map['velocity'] as num).toDouble(),
    );
  }

  /// MIDI note number equivalent (0–127).
  int get midiNote => key + 21;

  @override
  String toString() =>
      'RawNoteData(key: $key, frame: ${frame.toStringAsFixed(3)}, '
      'onset: ${onset.toStringAsFixed(3)}, offset: ${offset.toStringAsFixed(3)}, '
      'velocity: ${velocity.toStringAsFixed(3)})';
}
