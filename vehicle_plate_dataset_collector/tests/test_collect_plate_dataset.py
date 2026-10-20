import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "collect_plate_dataset.py"
SPEC = importlib.util.spec_from_file_location("collector", SCRIPT_PATH)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "output_root": "dataset",
            "timeout_seconds": 1,
            "cameras": {
                "left": {"name": "camera1", "snapshot_url": "http://camera1/snapshot"},
                "right": {"name": "camera2", "snapshot_url": "http://camera2/snapshot"},
            },
        }
        self.session = object()
        self.captured_sides = set()

    def test_one_snapshot_for_repeated_allowed_status(self):
        item = {"left": collector.ENTRY_ALLOWED, "right": "Ожидание транспорта", "car": "А575ОЕ51"}
        with patch.object(collector, "save_event") as save_event:
            collector.process_item(item, self.config, self.captured_sides, self.session)
            collector.process_item(item, self.config, self.captured_sides, self.session)

        self.assertEqual(save_event.call_count, 1)
        self.assertEqual(save_event.call_args.args[0:2], ("left", "А575ОЕ51"))

    def test_status_change_rearms_side_and_both_sides_are_saved(self):
        with patch.object(collector, "save_event") as save_event:
            collector.process_item(
                {"left": collector.ENTRY_ALLOWED, "right": collector.ENTRY_ALLOWED, "car": "А575ОЕ51"},
                self.config,
                self.captured_sides,
                self.session,
            )
            collector.process_item(
                {"left": "Ожидание транспорта", "right": "Ожидание транспорта", "car": ""},
                self.config,
                self.captured_sides,
                self.session,
            )
            collector.process_item(
                {"left": collector.ENTRY_ALLOWED, "right": "Ожидание транспорта", "car": "В123ВС51"},
                self.config,
                self.captured_sides,
                self.session,
            )

        self.assertEqual(save_event.call_count, 3)
        self.assertEqual([call.args[0] for call in save_event.call_args_list], ["left", "right", "left"])


if __name__ == "__main__":
    unittest.main()
