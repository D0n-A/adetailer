from __future__ import annotations

import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from rich import print  # noqa: A004  Shadowing built-in 'print'
from torchvision.transforms.functional import to_pil_image

REPO_ID = "Bingsu/adetailer"

DEFAULT_PT_MODELS = (
    "face_yolov8n.pt",
    "face_yolov8s.pt",
    "hand_yolov8n.pt",
    "person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt",
    "yolov8x-worldv2.pt",
)

DEFAULT_MEDIAPIPE_MODELS = {
    "mediapipe_face_full": "mediapipe_face_full",
    "mediapipe_face_short": "mediapipe_face_short",
    "mediapipe_face_mesh": "mediapipe_face_mesh",
    "mediapipe_face_mesh_eyes_only": "mediapipe_face_mesh_eyes_only",
}

T = TypeVar("T", int, float)


@dataclass
class PredictOutput(Generic[T]):
    bboxes: list[list[T]] = field(default_factory=list)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def hf_download(file: str, repo_id: str = REPO_ID, check_remote: bool = True) -> str:
    if check_remote:
        with suppress(Exception):
            return hf_hub_download(repo_id, file, etag_timeout=1)

        with suppress(Exception):
            return hf_hub_download(
                repo_id, file, etag_timeout=1, endpoint="https://hf-mirror.com"
            )

    with suppress(Exception):
        return hf_hub_download(repo_id, file, local_files_only=True)

    if check_remote:
        msg = f"[-] ADetailer: Failed to load model {file!r} from huggingface"
        print(msg)
    return "INVALID"


def safe_mkdir(path: str | os.PathLike[str]) -> None:
    path = Path(path)
    if not path.exists() and path.parent.exists() and os.access(path.parent, os.W_OK):
        path.mkdir()


def scan_model_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix == ".pt"]


def download_models(*names: str, check_remote: bool = True) -> dict[str, str]:
    models = OrderedDict()
    with ThreadPoolExecutor() as executor:
        for name in names:
            if "-world" in name:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    repo_id="Bingsu/yolo-world-mirror",
                    check_remote=check_remote,
                )
            else:
                models[name] = executor.submit(
                    hf_download,
                    name,
                    check_remote=check_remote,
                )
    return {name: future.result() for name, future in models.items()}


def get_models(
    *dirs: str | os.PathLike[str],
    huggingface: bool = True,
    extra_key_mode: str = "legacy",
    include_conflicting_models: bool = False,
    warn_on_skipped: bool = False,
) -> OrderedDict[str, str]:
    models = OrderedDict()
    models.update(download_models(*DEFAULT_PT_MODELS, check_remote=huggingface))

    models.update(DEFAULT_MEDIAPIPE_MODELS)

    invalid_keys = [k for k, v in models.items() if v == "INVALID"]
    for key in invalid_keys:
        models.pop(key)

    scan_dirs: list[Path] = []
    for dir_ in dirs:
        if not dir_:
            continue

        dir_str = str(dir_).strip()
        if not dir_str:
            continue
        scan_dirs.append(Path(dir_str))

    skipped: list[str] = []

    def extra_namespace(i: int) -> str:
        # i is 1-based index within extra dirs
        return "extra" if i == 1 else f"extra{i}"

    for dir_idx, base_dir in enumerate(scan_dirs):
        is_extra = dir_idx > 0
        extra_idx = dir_idx  # 1..N for extra dirs

        if not base_dir.is_dir():
            if warn_on_skipped:
                skipped.append(f"dir not found: {base_dir}")
            continue

        # Collect root models (always), then add subfolder models if they exist.
        root_files = [p for p in base_dir.glob("*.pt") if p.is_file()]
        root_files.sort(key=lambda p: p.name)

        subdir_files = (
            [p for p in scan_model_dir(base_dir) if p.parent != base_dir]
            if any(item.is_dir() for item in base_dir.iterdir())
            else []
        )
        subdir_files.sort(key=lambda p: p.relative_to(base_dir).as_posix())

        has_subdir_models = bool(subdir_files)
        model_paths = [*root_files, *subdir_files] if has_subdir_models else root_files

        for path in model_paths:
            rel_key = (
                path.relative_to(base_dir).as_posix()
                if has_subdir_models and path.parent != base_dir
                else path.name
            )

            key = rel_key

            if is_extra:
                mode = (extra_key_mode or "legacy").lower()
                ns = extra_namespace(extra_idx)
                if mode == "prefix":
                    key = f"{ns}/{rel_key}"
                elif mode == "auto" and key in models:
                    key = f"{ns}/{rel_key}"

            if key in models:
                if include_conflicting_models and not is_extra:
                    conflict_key = f"local/{rel_key}"
                    if conflict_key in models:
                        if warn_on_skipped:
                            skipped.append(f"duplicate key: {conflict_key} -> {path}")
                        continue
                    key = conflict_key
                else:
                    if warn_on_skipped:
                        skipped.append(f"duplicate key: {key} -> {path}")
                    continue
            models[key] = str(path)

    if warn_on_skipped and skipped:
        print("[-] ADetailer: Skipped models while scanning:", file=sys.stderr)
        for line in skipped:
            print(f"  - {line}", file=sys.stderr)

    return models


def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[list[int]]:
    """
    Parameters
    ----------
        masks: list[Image.Image]
            A list of masks
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        bboxes: list[list[float]]
        A list of bounding boxes

    """
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)  # noqa: PLW2901
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes


def ensure_pil_image(image: Any, mode: str = "RGB") -> Image.Image:
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != mode:
        image = image.convert(mode)
    return image
