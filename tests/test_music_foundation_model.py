from mantra.music_foundation_model import MusicFoundationModel
from mantra.intelligence.music_foundation_model import MusicFoundationModel as IntelligenceMusicFoundationModel


def test_music_foundation_model_embeddings_and_train():
    model = MusicFoundationModel(dim=64)
    emb = model.build_multimodal_embedding(
        audio="track.wav",
        lyrics="ambient breeze",
        metadata={"genre": "ambient"},
        artist_id="artist-1",
        artist_neighbors=["artist-2"],
    )
    assert emb.shape[0] == 64
    status = model.train_or_finetune([{"audio": "a"}])
    assert status["trained"] is True


def test_intelligence_music_foundation_model_embed_and_structure():
    model = IntelligenceMusicFoundationModel(embedding_dim=64)
    audio_vec = model.embed_audio("demo.wav")
    midi_vec = model.embed_midi("demo.mid")
    feat_vec = model.embed_features([0.1, 0.2, 0.3])
    structure = model.analyze_structure("demo.wav")
    assert audio_vec.shape[0] == 64
    assert midi_vec.shape[0] == 64
    assert feat_vec.shape[0] == 64
    assert "tempo" in structure
    assert "key" in structure

