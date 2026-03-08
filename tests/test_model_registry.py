from pathlib import Path
from uuid import uuid4

from mantra.model_registry import ModelRegistry


def test_model_registry_register_latest_active():
    root = Path(f"test_model_registry_{uuid4().hex}")
    root.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(root_dir=str(root))
    model_file = root / "dummy.bin"
    model_file.write_text("x", encoding="utf-8")

    registry.register("reranker", "v1", str(model_file))
    registry.register("reranker", "v2", str(model_file))

    latest = registry.get_latest("reranker")
    assert latest is not None
    assert latest["model_name"] == "reranker"

    registry.set_active("reranker", "v1")
    active = registry.get_active("reranker")
    assert active is not None
    assert active["version"] == "v1"
