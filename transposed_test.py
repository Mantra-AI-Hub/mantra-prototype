from mido import MidiFile

original = MidiFile("test_midis/test.mid")

transposed = MidiFile()

for track in original.tracks:
    for msg in track:
        if msg.type == "note_on" or msg.type == "note_off":
            msg.note += 2
    transposed.tracks.append(track)

transposed.save("test_midis/test_transposed.mid")

print("Transposed MIDI created")