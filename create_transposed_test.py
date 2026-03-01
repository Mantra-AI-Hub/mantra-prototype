from mido import MidiFile, MidiTrack, Message

# Загружаем оригинал
original = MidiFile("test.mid")

transposed = MidiFile()
track = MidiTrack()
transposed.tracks.append(track)

SHIFT = 5  # транспонируем на 5 полутонов

for msg in original.tracks[0]:
    if msg.type == "note_on" or msg.type == "note_off":
        msg = msg.copy(note=msg.note + SHIFT)
    track.append(msg)

transposed.save("test_transposed.mid")

print("Created test_transposed.mid")