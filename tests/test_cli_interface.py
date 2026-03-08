import json
from pathlib import Path
import uuid

from mantra.interfaces.cli.mantra_cli import main


def test_cli_index_and_search(capsys):
    db_path = Path(f"test_cli_{uuid.uuid4().hex}.db")
    dataset = Path("test_midis")
    midi_a = dataset / "test.mid"

    rc_index = main(["index", str(dataset), "--db-path", str(db_path)])
    index_out = capsys.readouterr().out
    assert rc_index == 0
    assert "Indexed 1 MIDI files" in index_out

    rc_search = main(["search", str(midi_a), "--db-path", str(db_path)])
    search_out = capsys.readouterr().out
    assert rc_search == 0
    assert "Search results:" in search_out
    assert "test.mid" in search_out


def test_cli_explain(capsys):
    rc = main(["explain", "test.mid", "test_transposed.mid"])
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert "similarity_score" in payload
    assert "pitch_similarity" in payload
