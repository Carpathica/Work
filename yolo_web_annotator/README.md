# YOLO Web Annotator

Browser-based image annotation tool with YOLO auto-labeling and YOLO-format export.

## Features

- Load local dataset folder.
- Connect your pretrained YOLO model (`.pt`, `.onnx`) for auto-labeling.
- Built-in path picker for dataset/model/classes/labels folders.
- Load classes from YAML/TXT or from manual list.
- Edit boxes: add, move, resize, delete.
- `Save` current image and `Save All` for all unsaved images.
- Optional overlap cleanup with configurable threshold (%), removes lower-confidence boxes.
- Keyboard shortcuts and image switching by arrow keys.
- Zoom in/out/fit and mouse wheel zoom.
- Right-click class menu on a box.
- Save in YOLO TXT format.

## Dataset layout

Supported layouts:

- `dataset/images/...` with `dataset/labels/...`
- flat image folder
- custom labels folder path (optional, can be outside dataset)

## Install

```powershell
cd yolo_web_annotator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
cd yolo_web_annotator
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Quick use

1. Set dataset folder.
2. Optionally set model path.
3. Optionally set classes YAML/TXT or classes list.
4. Optionally set output labels folder.
5. Click `Load Dataset`.
6. Use `Auto-label`, edit boxes, then `Save` or `Save All`.

## Hotkeys

- `Ctrl+S`: save current image
- `Ctrl+Shift+S`: save all unsaved images
- `Ctrl+Z`: undo last box operation
- `P`: auto-label current image
- `Left/Right`: previous/next image
- `Shift + Arrow`: move selected box
- `Delete`: delete selected box
- `+` / `-`: zoom
- `0`: fit zoom
