from mido import Message, MidiFile, MidiTrack

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

notes = [60, 62, 64, 65, 67, 69, 71, 72]

for note in notes:
    track.append(Message('note_on', note=note, velocity=64, time=0))
    track.append(Message('note_off', note=note, velocity=64, time=200))

mid.save("test_midis/test.mid")

print("Test MIDI created: test_midis/test.mid")