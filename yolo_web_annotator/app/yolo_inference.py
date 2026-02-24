from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .schemas import Box

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency import errors
    YOLO = None  # type: ignore[assignment]


class YoloRunner:
    def __init__(self) -> None:
        self._model = None
        self._model_path: Path | None = None

    def predict(
        self,
        image_path: Path,
        model_path: Path,
        conf: float = 0.25,
    ) -> Tuple[List[Box], Dict[int, str]]:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install dependencies from yolo_web_annotator/requirements.txt"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if self._model is None or self._model_path != model_path:
            self._model = YOLO(str(model_path))
            self._model_path = model_path

        results = self._model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
        )
        if not results:
            return [], {}

        result = results[0]
        names = getattr(result, "names", {}) or {}
        class_names = {int(key): str(value) for key, value in names.items()}

        boxes: List[Box] = []
        for item in result.boxes:
            xyxy = item.xyxy[0].tolist()
            class_id = int(item.cls[0].item()) if item.cls is not None else 0
            score = float(item.conf[0].item()) if item.conf is not None else None
            x1, y1, x2, y2 = xyxy
            boxes.append(
                Box(
                    class_id=max(class_id, 0),
                    x=float(x1),
                    y=float(y1),
                    width=max(0.0, float(x2 - x1)),
                    height=max(0.0, float(y2 - y1)),
                    score=score,
                )
            )
        return boxes, class_names
