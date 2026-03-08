from mantra.generative_music_lab import GenerativeMusicLab


def test_generative_music_lab_combines_generators():
    lab = GenerativeMusicLab()
    lab.register_generator("g1", lambda prompt: {"prompt": prompt, "notes": [60, 62]})
    lab.register_generator("g2", lambda prompt: {"prompt": prompt, "notes": [64, 65]})
    result = lab.combine_multiple_generators("chill")
    assert result["count"] == 2
