import mido
from mido import Message, MidiFile, MidiTrack


def create_midi(filename, notes):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=0, time=0))

    for note in notes:
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=480))

    mid.save(filename)
    print(f"Created {filename}")


# Оригинальная мелодия (C major)
melody1 = [60, 62, 64, 65, 67, 69, 71, 72]

# Немного изменённая версия
melody2 = [60, 62, 64, 66, 67, 69, 71, 72]

create_midi("melody1.mid", melody1)
create_midi("melody2.mid", melody2)
