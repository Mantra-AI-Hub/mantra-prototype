import mido
from mido import MidiFile, MidiTrack, Message

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Простая гамма C major
notes = [60, 62, 64, 65, 67, 69, 71, 72]

for note in notes:
    track.append(Message("note_on", note=note, velocity=64, time=0))
    track.append(Message("note_off", note=note, velocity=64, time=480))

mid.save("test.mid")

print("test.mid created")
