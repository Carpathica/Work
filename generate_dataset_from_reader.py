#!/usr/bin/env python3
"""
Интерактивный скрипт для генерации датасета в формате YOK1_dataset путём запуска read_container.py.

Отображает изображение с результатом распознавания и позволяет подтвердить (Y) или отклонить (N).

Формат выходного dataset.jsonl:
{
    "id": "camera1_01-17-02",
    "image": "images/camera1_01-17-02.jpg",
    "scenario": "extract_number_general",
    "expected": {
        "owner_code": "TGHU",
        "registration_number": "095205",
        "check_digit": "0",
        "type_size_code": ""
    },
    "metadata": {
        "scenario_version": 1,
        "iso_code": "TGHU0952050",
        "full_code": "TGHU0952050",
        "source_image": "C:\\path\\to\\source.jpg",
        "model": "C:\\path\\to\\model.pt",
        "latency_ms": 472.778,
        "raw_detections_count": 11,
        "detections_count": 11,
        "duplicates_removed": 0
    }
}

Управление:
- Y / y / Enter / Space — сохранить (распознано правильно)
- N / n — пропустить (распознано неправильно)
- Q / q / Esc — выйти
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import cv2
    HAS_CV = True
except ImportError:
    HAS_CV = False

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Интерактивная генерация датасета в формате YOK1_dataset через запуск read_container.py"
    )
    parser.add_argument(
        "--reader",
        required=True,
        type=Path,
        help="Путь к read_container.py"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Путь к YOLO модели (.pt)"
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Папка с изображениями или glob-паттерн"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Выходная папка для датасета"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Рекурсивный поиск изображений"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Порог уверенности YOLO (по умолчанию 0.15 как в read_container.py)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Порог NMS IoU"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Максимум детекций на изображение"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.35,
        help="Порог слияния дублей (0 для отключения)"
    )
    parser.add_argument(
        "--scenario-name",
        default="extract_number_general",
        help="Имя сценария"
    )
    parser.add_argument(
        "--scenario-version",
        type=int,
        default=1,
        help="Версия сценария"
    )
    parser.add_argument(
        "--dataset-file",
        default="dataset.jsonl",
        help="Имя файла датасета"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Копировать изображения в output_dir/images/"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Ограничение количества изображений"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Неинтерактивный режим: сохранять все результаты автоматически"
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.0,
        help="Масштаб окна предпросмотра (по умолчанию 1.0)"
    )
    return parser.parse_args()


def collect_images(source: Path, recursive: bool = False) -> List[Path]:
    """Собрать список изображений из папки."""
    if source.is_file():
        return [source.resolve()] if source.suffix.lower() in IMAGE_SUFFIXES else []
    
    if source.is_dir():
        if recursive:
            return sorted(
                p.resolve() for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        return sorted(
            p.resolve() for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    
    # Glob паттерн
    if "*" in str(source):
        return sorted(p.resolve() for p in Path(".").glob(str(source)) if p.suffix.lower() in IMAGE_SUFFIXES)
    
    return []


def run_reader(
    reader_path: Path,
    model_path: Path,
    image_path: Path,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: Optional[float]
) -> Optional[Dict[str, Any]]:
    """Запустить read_container.py для одного изображения и вернуть результат."""
    cmd = [
        sys.executable,
        str(reader_path),
        "--weights", str(model_path),
        "--source", str(image_path),
        "--conf", str(conf),
        "--iou", str(iou),
        "--max-det", str(max_det),
    ]
    
    if merge_iou is not None and merge_iou > 0:
        cmd.extend(["--merge-iou", str(merge_iou)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(reader_path.parent)
        )
        
        if result.returncode != 0:
            print(f"[error] read_container.py вернул код {result.returncode}: {result.stderr.strip()}")
            return None
        
        output = result.stdout.strip()
        if not output:
            return None
        
        # read_container.py выводит JSON строку
        data = json.loads(output)
        return data
        
    except subprocess.TimeoutExpired:
        print(f"[warn] таймаут для {image_path.name}")
        return None
    except json.JSONDecodeError as e:
        print(f"[warn] не удалось распарсить JSON вывод: {e}")
        return None
    except Exception as e:
        print(f"[error] ошибка при запуске: {e}")
        return None


def parse_reader_result(data: Dict[str, Any]) -> Dict[str, str]:
    """Преобразовать результат read_container.py в формат expected полей."""
    result_list = data.get("result", [])
    
    fields = {
        "owner_code": "",
        "registration_number": "",
        "check_digit": "",
        "type_size_code": ""
    }
    
    for item in result_list:
        if isinstance(item, dict):
            label = item.get("label", "")
            text = item.get("text", "")
            if label in fields:
                fields[label] = text
    
    return fields


def build_iso_code(expected: Dict[str, str]) -> str:
    """Построить ISO код из полей."""
    return (
        f"{expected.get('owner_code', '')}"
        f"{expected.get('registration_number', '')}"
        f"{expected.get('check_digit', '')}"
    )


def build_full_code(expected: Dict[str, str]) -> str:
    """Построить полный код с type_size_code."""
    iso = build_iso_code(expected)
    type_size = expected.get("type_size_code", "")
    if type_size:
        return f"{iso} {type_size}".strip()
    return iso.strip()


def save_case(
    source_image: Path,
    output_images_dir: Path,
    dataset_path: Path,
    case_id: str,
    scenario_name: str,
    scenario_version: int,
    expected: Dict[str, str],
    metadata: Dict[str, Any],
    copy_images: bool = False
) -> None:
    """Сохранить кейс в датасет."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    if copy_images:
        suffix = source_image.suffix.lower() or ".jpg"
        out_image_name = f"{case_id}{suffix}"
        out_image_path = output_images_dir / out_image_name
        shutil.copy2(source_image, out_image_path)
        image_ref = f"images/{out_image_name}"
    else:
        # Используем абсолютный путь к исходному изображению
        image_ref = str(source_image.resolve())
    
    iso_code = build_iso_code(expected)
    full_code = build_full_code(expected)
    
    row = {
        "id": case_id,
        "image": image_ref,
        "scenario": scenario_name,
        "expected": expected,
        "metadata": {
            "scenario_version": scenario_version,
            "iso_code": iso_code,
            "full_code": full_code,
            **metadata,
        },
    }
    
    with dataset_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_unique_case_id(stem: str, existing: Set[str]) -> str:
    """Убедиться, что ID уникален."""
    normalized = stem.strip() or "sample"
    if normalized not in existing:
        return normalized
    
    idx = 2
    while True:
        candidate = f"{normalized}_{idx}"
        if candidate not in existing:
            return candidate
        idx += 1


def load_existing_case_ids(dataset_path: Path) -> Set[str]:
    """Загрузить существующие ID кейсов."""
    if not dataset_path.exists():
        return set()
    
    ids: Set[str] = set()
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get("id")
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    
    return ids


def create_annotated_image(
    image_path: Path,
    result: Dict[str, Any],
    expected: Dict[str, str]
) -> Optional[Any]:
    """Создать изображение с аннотацией результата."""
    if not HAS_CV:
        return None
    
    # Загрузка изображения
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    
    # Получение размеров для масштабирования текста
    height, width = frame.shape[:2]
    font_scale = min(1.0, max(0.5, width / 800))
    thickness = max(1, int(2 * font_scale))
    
    # Извлечение результата распознавания
    result_list = result.get("result", [])
    recognized_text = ""
    for item in result_list:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                recognized_text += text
    
    elapsed_ms = result.get("elapsed_ms", 0)
    
    # Формирование строк для отображения
    iso_code = build_iso_code(expected)
    full_code = build_full_code(expected)
    type_size = expected.get("type_size_code", "")
    
    lines = [
        f"ISO: {iso_code}",
        f"Full: {full_code}",
        f"Type/Size: {type_size}",
        f"Time: {elapsed_ms:.1f}ms",
        "",
        "Y/Enter/Space = сохранить (правильно)",
        "N = пропустить (неправильно)",
        "Q/Esc = выход"
    ]
    
    # Рисование фона для текста
    text_height = 30 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, text_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Рисование текста
    y_offset = 25
    for i, line in enumerate(lines):
        color = (100, 255, 100) if i == 0 else (200, 255, 200) if i < 3 else (180, 180, 255)
        if i >= 4:  # Инструкции
            color = (200, 200, 255)
        cv2.putText(
            frame,
            line,
            (15, y_offset + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.7,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    return frame


def show_preview(image_path: Path, result: Dict[str, Any], expected: Dict[str, str]) -> Optional[bool]:
    """
    Показать предпросмотр с результатом распознавания.
    Возвращает True если сохранить, False если пропустить, None если выйти.
    """
    if not HAS_CV:
        print("[warn] OpenCV не установлен. Используйте --no-interactive для автоматического режима.")
        return None
    
    frame = create_annotated_image(image_path, result, expected)
    if frame is None:
        return None
    
    window_name = "Dataset Builder - Y/N для подтверждения"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    
    # Ожидание нажатия клавиши
    key = cv2.waitKey(0) & 0xFF
    
    cv2.destroyAllWindows()
    
    if key in (ord("q"), ord("Q"), 27):  # Q, q, Esc
        return None
    elif key in (ord("y"), ord("Y"), ord("n"), ord("N")):
        return key in (ord("y"), ord("Y"))
    elif key in (13, 32):  # Enter, Space
        return True
    else:
        return None


def main() -> int:
    args = parse_args()
    
    # Проверка зависимостей
    if not args.no_interactive and not HAS_CV:
        print("[error] OpenCV не установлен. Установите opencv-python или используйте --no-interactive")
        return 1
    
    # Проверка путей
    reader_path = args.reader.resolve()
    if not reader_path.exists():
        print(f"[error] read_container.py не найден: {reader_path}")
        return 1
    
    model_path = args.model.resolve()
    if not model_path.exists():
        print(f"[error] модель не найдена: {model_path}")
        return 1
    
    source_path = args.source.resolve()
    if not source_path.exists():
        print(f"[error] источник не найден: {source_path}")
        return 1
    
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_images_dir = output_dir / "images"
    dataset_path = output_dir / args.dataset_file
    
    # Сбор изображений
    image_paths = collect_images(source_path, recursive=args.recursive)
    if not image_paths:
        print(f"[error] изображения не найдены: {source_path}")
        return 1
    
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    
    print(f"[info] найдено {len(image_paths)} изображений")
    print(f"[info] модель: {model_path}")
    print(f"[info] вывод: {output_dir}")
    print(f"[info] режим: {'интерактивный' if not args.no_interactive else 'автоматический'}")
    
    if not args.no_interactive:
        print("[info] управление: Y=сохранить, N=пропустить, Q=выход")
    
    # Загрузка существующих ID
    existing_ids = load_existing_case_ids(dataset_path)
    
    processed = 0
    saved = 0
    skipped = 0
    errors = 0
    latencies_ms: List[float] = []
    
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] обработка {image_path.name}...")
        
        started = time.perf_counter()
        
        # Запуск read_container.py
        result = run_reader(
            reader_path=reader_path,
            model_path=model_path,
            image_path=image_path,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            merge_iou=args.merge_iou
        )
        
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        
        if result is None:
            errors += 1
            print(f"[warn] нет результата для {image_path.name}")
            continue
        
        processed += 1
        
        # Парсинг результата
        expected = parse_reader_result(result)
        iso = build_iso_code(expected)
        
        print(f"[result] ISO: {iso or '(пусто)'}, Type/Size: {expected.get('type_size_code', '')}")
        
        # Определение действия
        should_save = False
        
        if args.no_interactive:
            # Автоматический режим - сохранять всё
            should_save = True
            print("[auto] автоматическое сохранение")
        else:
            # Интерактивный режим
            decision = show_preview(image_path, result, expected)
            
            if decision is None:
                print("[quit] выход по запросу пользователя")
                break
            elif decision:
                should_save = True
                print("[save] подтверждено пользователем (Y)")
            else:
                skipped += 1
                print("[skip] отклонено пользователем (N)")
                continue
        
        if not should_save:
            continue
        
        # Генерация уникального ID
        case_id = ensure_unique_case_id(image_path.stem, existing_ids)
        existing_ids.add(case_id)
        
        # Получение количества детекций (если есть в выводе)
        raw_count = result.get("raw_detections_count", 0)
        det_count = result.get("detections_count", 0)
        duplicates_removed = result.get("duplicates_removed", 0)
        
        # Сохранение кейса
        save_case(
            source_image=image_path,
            output_images_dir=output_images_dir,
            dataset_path=dataset_path,
            case_id=case_id,
            scenario_name=args.scenario_name,
            scenario_version=args.scenario_version,
            expected=expected,
            metadata={
                "source_image": str(image_path),
                "model": str(model_path),
                "latency_ms": round(elapsed_ms, 3),
                "raw_detections_count": raw_count,
                "detections_count": det_count,
                "duplicates_removed": duplicates_removed,
            },
            copy_images=args.copy_images
        )
        
        saved += 1
        print(f"[save] {case_id} | ISO: {iso or '(пусто)'}")
    
    # Статистика
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    
    summary = {
        "reader": str(reader_path),
        "model": str(model_path),
        "source": str(source_path),
        "output_dir": str(output_dir),
        "dataset_file": str(dataset_path),
        "images_total": len(image_paths),
        "processed": processed,
        "saved": saved,
        "skipped": skipped,
        "errors": errors,
        "avg_latency_ms": round(avg_latency, 3),
        "copy_images": args.copy_images,
        "scenario": args.scenario_name,
        "scenario_version": args.scenario_version,
        "interactive": not args.no_interactive,
    }
    
    summary_path = output_dir / "dataset_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print("\n" + "=" * 50)
    print("[done] Генерация датасета завершена")
    print(f"[done] датасет: {dataset_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[stats] всего изображений: {len(image_paths)}")
    print(f"[stats] обработано: {processed}")
    print(f"[stats] сохранено: {saved}")
    print(f"[stats] пропущено: {skipped}")
    print(f"[stats] ошибок: {errors}")
    print(f"[stats] среднее время обработки: {avg_latency:.2f} мс")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())