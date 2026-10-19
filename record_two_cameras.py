"""Record two Axis MJPEG streams with OpenCV. Press F8 to start or stop."""

import json
import msvcrt
import threading
from datetime import datetime
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
RECORDINGS_DIR = SCRIPT_DIR / "recordings"


def record_camera(camera: dict, output_path: Path, fps: float, stop_event: threading.Event) -> None:
    capture = cv2.VideoCapture(camera["url"])
    writer = None

    try:
        while not stop_event.is_set():
            received, frame = capture.read()
            if not received:
                continue

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def start_recording(cameras: list[dict], fps: float) -> tuple[threading.Event, list[threading.Thread]]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stop_event = threading.Event()
    threads = []

    for camera in cameras:
        output_path = RECORDINGS_DIR / f'{camera["name"]}_{timestamp}.mp4'
        thread = threading.Thread(
            target=record_camera,
            args=(camera, output_path, fps, stop_event),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    return stop_event, threads


def stop_recording(stop_event: threading.Event, threads: list[threading.Thread]) -> None:
    stop_event.set()
    for thread in threads:
        thread.join()


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    RECORDINGS_DIR.mkdir(exist_ok=True)
    stop_event = None
    threads = []

    print("F8 - start/stop recording. Ctrl+C - exit.")

    try:
        while True:
            key = msvcrt.getwch()

            if key not in ("\x00", "\xe0") or msvcrt.getwch() != "B":
                continue

            if stop_event is None:
                stop_event, threads = start_recording(config["cameras"], config["fps"])
                print("Recording started.")
            else:
                stop_recording(stop_event, threads)
                stop_event = None
                threads = []
                print("Recording stopped.")
    except KeyboardInterrupt:
        if stop_event is not None:
            stop_recording(stop_event, threads)


if __name__ == "__main__":
    main()
