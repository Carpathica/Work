from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Box(BaseModel):
    class_id: int = Field(ge=0)
    x: float
    y: float
    width: float
    height: float
    score: Optional[float] = None


class SessionRequest(BaseModel):
    dataset_dir: str
    classes: List[str] = Field(default_factory=list)
    classes_file: Optional[str] = None
    model_path: Optional[str] = None
    labels_dir: Optional[str] = None


class SaveAnnotationsRequest(BaseModel):
    boxes: List[Box] = Field(default_factory=list)


class ImageAnnotations(BaseModel):
    path: str
    boxes: List[Box] = Field(default_factory=list)


class SaveAllRequest(BaseModel):
    items: List[ImageAnnotations] = Field(default_factory=list)


class PredictRequest(BaseModel):
    model_path: Optional[str] = None
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
