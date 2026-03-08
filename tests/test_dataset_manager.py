from pathlib import Path
from uuid import uuid4

from mantra.dataset_manager import DatasetManager


def test_dataset_manager_register_list_status():
    db_path = f"test_dataset_manager_{uuid4().hex}.db"
    manager = DatasetManager(db_path=db_path)

    created = manager.register_dataset("ds1", "My Dataset")
    assert created["dataset_id"] == "ds1"

    manager.set_status("ds1", "ingesting", track_count=123)
    one = manager.get_dataset("ds1")
    assert one is not None
    assert one["ingestion_status"] == "ingesting"
    assert one["track_count"] == 123

    listing = manager.list_datasets()
    assert listing and listing[0]["dataset_id"] == "ds1"
