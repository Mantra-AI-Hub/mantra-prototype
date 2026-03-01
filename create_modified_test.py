from mido import MidiFile, MidiTrack, Message

original = MidiFile("test.mid")

modified = MidiFile()
track = MidiTrack()
modified.tracks.append(track)

changed = False

for msg in original.tracks[0]:
    if msg.type == "note_on" and not changed:
        msg = msg.copy(note=msg.note + 3)  # меняем одну ноту
        changed = True
    track.append(msg)

modified.save("test_modified.mid")

print("Created test_modified.mid")