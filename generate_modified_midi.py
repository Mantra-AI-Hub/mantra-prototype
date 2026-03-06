from mido import MidiFile

original = MidiFile("test_midis/test.mid")

modified = MidiFile()

for track in original.tracks:
    new_track = []
    for msg in track:
        if msg.type == "note_on":
            msg.velocity = min(127, msg.velocity + 5)
        new_track.append(msg)
    modified.tracks.append(track)

modified.save("test_midis/test_modified.mid")

print("Modified MIDI created")