/// The type of a [NoteEvent].
enum NoteEventType {
  /// A note was pressed / started sounding.
  noteOn,

  /// A note was released / stopped sounding.
  noteOff,
}

/// A single piano note event detected from microphone audio.
///
/// Note numbers follow the **MIDI standard** (0–127):
/// - A0  = MIDI 21  (lowest standard piano key)
/// - C4  = MIDI 60  (middle C)
/// - C8  = MIDI 108 (highest standard piano key)
class NoteEvent {
  /// Whether this is a note-on or note-off event.
  final NoteEventType type;

  /// MIDI note number (0–127).
  final int note;

  /// Normalised velocity in the range 0.0–1.0.
  /// Always 0.0 for [NoteEventType.noteOff].
  final double velocity;

  const NoteEvent({
    required this.type,
    required this.note,
    required this.velocity,
  });

  factory NoteEvent.fromMap(Map<Object?, Object?> map) {
    final typeStr = map['type'] as String;
    return NoteEvent(
      type: typeStr == 'noteOn' ? NoteEventType.noteOn : NoteEventType.noteOff,
      note: (map['note'] as num).toInt(),
      velocity: (map['velocity'] as num).toDouble(),
    );
  }

  @override
  String toString() =>
      'NoteEvent(type: $type, note: $note, velocity: ${velocity.toStringAsFixed(2)})';
}
