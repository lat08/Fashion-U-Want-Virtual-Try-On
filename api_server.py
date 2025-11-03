"""
Virtual Try-On API Server
Hệ thống API xử lý đồng thời nhiều request từ người dùng
"""

import os
import sys
import uuid
import shutil
import cv2
import json
import copy
import subprocess
import mimetypes
import numpy as np
from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import OpenPose PyTorch
sys.path.insert(0, str(Path("OpenPose.PyTorch").absolute()))
from src.body import Body
from src import util

# Import các module xử lý
from clothseg import process_images as process_cloth_segmentation
from PIL import Image, ImageDraw
from get_parse_agnostic import get_im_parse_agnostic
from get_agnostic2 import get_img_agnostic


# =========================
# Configuration
# =========================
class Config:
    BASE_DIR = Path(__file__).parent
    OPENPOSE_MODEL_PATH = BASE_DIR / "OpenPose.PyTorch" / "model" / "pose_iter_584000.caffemodel.pt"
    TEMP_DIR = BASE_DIR / "temp_sessions"
    MAX_WORKERS = 4  # Số lượng request xử lý đồng thời
    SESSION_CLEANUP_HOURS = 24  # Xóa session sau 24 giờ


config = Config()
config.TEMP_DIR.mkdir(exist_ok=True)

# Thread pool để xử lý đồng thời
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Khởi tạo OpenPose model (global để tái sử dụng)
print("🔧 Loading OpenPose model...")
body_estimation = Body(str(config.OPENPOSE_MODEL_PATH), 'body25')
print("✅ OpenPose model loaded")


# =========================
# FastAPI App
# =========================
tags_metadata = [
    {
        "name": "Health",
        "description": "Service status and readiness probes.",
    },
    {
        "name": "Upload",
        "description": "Upload model and cloth images to create a processing session.",
    },
    {
        "name": "Process",
        "description": "Run the full virtual try-on pipeline for a session.",
    },
    {
        "name": "Result",
        "description": "Fetch the generated try-on image(s).",
    },
    {
        "name": "Progress",
        "description": "Realtime pipeline progress over WebSocket.",
    },
    {
        "name": "Sessions",
        "description": "Manage session lifecycle and cleanup.",
    },
]

app = FastAPI(
    title="Virtual Try-On API",
    description="REST API for virtual clothing try-on using HR-VITON, OpenPose, Graphonomy, and DensePose.",
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "defaultModelsExpandDepth": -1,
    },
)

# CORS để cho phép web truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Constants & Helpers
# =========================
PROGRESS_QUEUES = {}

def _safe_progress_emit(progress_cb, event: str, data: dict = None):
    if progress_cb is None:
        return
    try:
        payload = {"event": event}
        if data:
            payload.update(data)
        progress_cb(payload)
    except Exception:
        pass


def build_content_url(file_path: Path) -> str:
    """Build a /content URL for a given file path, supporting Colab and local modes."""
    # Determine base path exposed via /content endpoint
    if Path("/content").exists():
        base_path = Path("/content")
    else:
        base_path = config.BASE_DIR

    try:
        relative_path = file_path.resolve().relative_to(base_path.resolve())
    except Exception:
        # Fallback to best-effort relative path
        try:
            relative_path = file_path.resolve().relative_to(config.BASE_DIR.resolve())
        except Exception:
            relative_path = Path(str(file_path))

    # Normalize separators for URLs
    return f"/content/{str(relative_path).replace(chr(92), '/')}"

PARSE_COLOR_TO_LABEL = {
    (0, 0, 0): 0,
    (255, 0, 0): 2,
    (0, 0, 255): 13,
    (85, 51, 0): 10,
    (255, 85, 0): 5,
    (0, 255, 255): 15,
    (51, 170, 221): 14,
    (0, 85, 85): 9,
    (0, 0, 85): 6,
    (0, 128, 0): 12,
    (177, 255, 85): 17,
    (85, 255, 170): 16,
    (0, 119, 221): 5,
}

DENSEPOSE_COLORMAP = {
    2: [20, 80, 194],
    3: [4, 98, 224],
    4: [8, 110, 221],
    9: [6, 166, 198],
    10: [22, 173, 184],
    15: [145, 191, 116],
    16: [170, 190, 105],
    17: [191, 188, 97],
    18: [216, 187, 87],
    19: [228, 191, 74],
    20: [240, 198, 60],
    21: [252, 205, 47],
    22: [250, 220, 36],
    23: [251, 235, 25],
    24: [248, 251, 14],
}


def run_subprocess(command, cwd=None):
    """Execute external command with error handling"""
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as exc:
        print(f"❌ Command failed with exit code {exc.returncode}")
        if exc.stderr:
            print(f"   stderr: {exc.stderr[:500]}")
        raise


def generate_image_parse_v3(segmentation_path: Path, output_dir: Path) -> Path:
    """Convert Graphonomy RGB segmentation map to grayscale label map."""
    if not segmentation_path.exists():
        raise ValueError(f"Segmentation image not found at: {segmentation_path}")

    img = Image.open(segmentation_path).convert("RGB")
    img_array = np.array(img)
    h, w, _ = img_array.shape
    gray_img = np.zeros((h, w), dtype=np.uint8)

    for color, label in PARSE_COLOR_TO_LABEL.items():
        mask = np.all(img_array == color, axis=-1)
        gray_img[mask] = label

    resized = cv2.resize(gray_img, (768, 1024), interpolation=cv2.INTER_NEAREST)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "00001_00.png"
    Image.fromarray(resized, "L").save(output_path)
    return output_path


def generate_densepose_outputs(session_dir: Path) -> Path:
    """Generate DensePose visualization matching get_densepose.py behavior."""
    model_path = session_dir / "model.jpg"
    data_json_path = session_dir / "data.json"

    if not model_path.exists():
        raise ValueError(f"Model image missing for DensePose: {model_path}")
    if not data_json_path.exists():
        raise ValueError(f"DensePose data.json missing at: {data_json_path}")

    img = Image.open(model_path).convert("RGB")
    img_w, img_h = img.size

    with open(data_json_path, "r") as f:
        json_data = json.load(f)

    if len(json_data) < 3:
        raise ValueError("DensePose JSON does not contain expected data")

    segmentation_data = np.array(json_data[0])
    seg_img = np.zeros((*segmentation_data.shape, 3), dtype=np.uint8)
    for key, color in DENSEPOSE_COLORMAP.items():
        seg_img[segmentation_data == key] = color

    box = list(json_data[2])
    if len(box) != 4:
        raise ValueError("DensePose bounding box has unexpected format")

    x1, y1, x2, y2 = box
    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)

    bg = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if w > 0 and h > 0:
        seg_h, seg_w = seg_img.shape[:2]
        crop_h = min(h, seg_h)
        crop_w = min(w, seg_w)
        x0 = max(0, x)
        y0 = max(0, y)
        x1_clip = min(img_w, x0 + crop_w)
        y1_clip = min(img_h, y0 + crop_h)
        bg[y0:y1_clip, x0:x1_clip, :] = seg_img[: y1_clip - y0, : x1_clip - x0, :]

    output_dir = session_dir / "HR-VITON" / "test" / "test" / "image-densepose"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "00001_00.jpg"
    Image.fromarray(bg, "RGB").save(output_path)
    return output_path


def generate_parse_agnostic_outputs(session_dir: Path):
    """Recreate get_parse_agnostic.py logic using session-scoped paths."""
    data_path = session_dir / "HR-VITON" / "test" / "test"
    image_dir = data_path / "image"
    parse_dir = data_path / "image-parse-v3"
    openpose_dir = data_path / "openpose_json"
    output_dir = data_path / "image-parse-agnostic-v3.2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists() or not parse_dir.exists() or not openpose_dir.exists():
        return

    for image_path in sorted(image_dir.glob("*.jpg")):
        pose_path = openpose_dir / image_path.name.replace(".jpg", "_keypoints.json")
        parse_path = parse_dir / image_path.name.replace(".jpg", ".png")

        if not pose_path.exists() or not parse_path.exists():
            continue

        with open(pose_path, "r") as f:
            pose_label = json.load(f)

        people = pose_label.get("people", [])
        if not people:
            continue

        pose_values = people[0].get("pose_keypoints_2d", [])
        if not pose_values:
            continue

        pose_data = np.array(pose_values).reshape((-1, 3))[:, :2]
        im_parse = Image.open(parse_path)
        agnostic = get_im_parse_agnostic(im_parse, pose_data)
        agnostic.save(output_dir / parse_path.name)


def generate_human_agnostic_outputs(session_dir: Path):
    """Recreate get_agnostic2.py logic using session-scoped paths."""
    data_path = session_dir / "HR-VITON" / "test" / "test"
    image_dir = data_path / "image"
    parse_dir = data_path / "image-parse-v3"
    openpose_dir = data_path / "openpose_json"
    output_dir = data_path / "agnostic-v3.2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists() or not parse_dir.exists() or not openpose_dir.exists():
        return

    for image_path in sorted(image_dir.glob("*.jpg")):
        pose_path = openpose_dir / image_path.name.replace(".jpg", "_keypoints.json")
        parse_path = parse_dir / image_path.name.replace(".jpg", ".png")

        if not pose_path.exists() or not parse_path.exists():
            continue

        with open(pose_path, "r") as f:
            pose_label = json.load(f)

        people = pose_label.get("people", [])
        if not people:
            continue

        pose_values = people[0].get("pose_keypoints_2d", [])
        if not pose_values:
            continue

        pose_data = np.array(pose_values).reshape((-1, 3))[:, :2]
        im = Image.open(image_path).convert("RGB")
        im_parse = Image.open(parse_path)
        agnostic = get_img_agnostic(im, im_parse, pose_data.copy())
        agnostic.save(output_dir / image_path.name)


# =========================
# Models
# =========================
class TryOnRequest(BaseModel):
    session_id: str
    background: bool = True


class TryOnResponse(BaseModel):
    session_id: str
    status: str
    message: str
    result_image_url: Optional[str] = None


class UploadResponse(BaseModel):
    session_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    openpose_model_loaded: bool


class DeleteSessionResponse(BaseModel):
    session_id: str
    status: str
    message: str


# =========================
# Helper Functions
# =========================
def create_session_dir(session_id: str) -> Path:
    """Tạo thư mục riêng cho mỗi session"""
    session_dir = config.TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Tạo các thư mục con cần thiết
    (session_dir / "input").mkdir(exist_ok=True)
    (session_dir / "output").mkdir(exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "image").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "cloth").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "cloth-mask").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "openpose_img").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "openpose_json").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "image-parse-v3").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "image-parse-agnostic-v3.2").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "agnostic-v3.2").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "test" / "test" / "image-densepose").mkdir(parents=True, exist_ok=True)
    (session_dir / "HR-VITON" / "Output").mkdir(parents=True, exist_ok=True)
    
    return session_dir

# work
def process_openpose(image_path: Path, session_dir: Path) -> dict:
    """
    Xử lý OpenPose sử dụng PyTorch (thay thế OpenPose C++)
    """
    print(f"🔍 Processing OpenPose for session: {session_dir.name}")
    print(f"   Image path: {image_path}")
    print(f"   Image exists: {image_path.exists()}")
    
    if not image_path.exists():
        raise ValueError(f"Image not found at: {image_path}")
    
    # Đọc ảnh
    oriImg = cv2.imread(str(image_path))
    if oriImg is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    print(f"   Image shape: {oriImg.shape}")
    
    # Resize về 768x1024
    if oriImg.shape[:2] != (1024, 768):
        print(f"   Resizing from {oriImg.shape[:2]} to (1024, 768)")
        oriImg = cv2.resize(oriImg, (768, 1024))
    else:
        print(f"   Already correct size")
    
    # Detect pose
    candidate, subset = body_estimation(oriImg)
    
    if len(subset) == 0:
        raise ValueError("No person detected in image")
    
    # Tạo output directories
    openpose_img_dir = session_dir / "HR-VITON" / "test" / "test" / "openpose_img"
    openpose_json_dir = session_dir / "HR-VITON" / "test" / "test" / "openpose_json"
    
    # Vẽ skeleton
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset, 'body25')
    
    # Lưu visualization
    output_img_path = openpose_img_dir / "00001_00_rendered.png"
    cv2.imwrite(str(output_img_path), canvas)
    
    # Convert sang JSON format
    result = {
        "version": 1.3,
        "people": []
    }
    
    for person in subset:
        person_keypoints = []
        for i in range(25):  # Body25 có 25 keypoints
            idx = int(person[i])
            if idx == -1:
                person_keypoints.extend([0, 0, 0])
            else:
                x, y, score = candidate[idx][:3]
                person_keypoints.extend([float(x), float(y), float(score)])
        
        result["people"].append({
            "person_id": [-1],
            "pose_keypoints_2d": person_keypoints,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        })
    
    # Lưu JSON
    output_json_path = openpose_json_dir / "00001_00_keypoints.json"
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ OpenPose completed: {len(result['people'])} person(s) detected")
    return result


# work
def process_cloth_mask(session_dir: Path):
    """Xử lý cloth segmentation"""
    print(f"👔 Processing cloth segmentation for session: {session_dir.name}")
    
    input_path = str(session_dir / "input" / "cloth.jpg")
    cloth_output = str(session_dir / "HR-VITON" / "test" / "test" / "cloth")
    mask_output = str(session_dir / "HR-VITON" / "test" / "test" / "cloth-mask")
    
    print(f"   Input: {input_path}")
    print(f"   Input exists: {Path(input_path).exists()}")
    print(f"   Cloth output: {cloth_output}")
    print(f"   Mask output: {mask_output}")
    
    if not Path(input_path).exists():
        raise ValueError(f"Cloth image not found at: {input_path}")
    
    process_cloth_segmentation(
        input_path=input_path,
        cloth_output_path=cloth_output,
        mask_output_path=mask_output
    )
    print("✅ Cloth segmentation completed")


def process_segmentation(session_dir: Path):
    """Xử lý Graphonomy semantic segmentation"""
    print(f"🎨 Processing semantic segmentation for session: {session_dir.name}")
    
    # Use the resized_img.jpg that was created earlier (384x512)
    resized_path = session_dir / "resized_img.jpg"
    print(f"   Resized image path: {resized_path}")
    print(f"   Resized image exists: {resized_path.exists()}")
    
    if not resized_path.exists():
        # Fallback: create it if not exists
        print(f"   WARNING: resized_img.jpg not found, creating it...")
        model_img = cv2.imread(str(session_dir / "input" / "model.jpg"))
        if model_img is None:
            raise ValueError("Cannot read model image for segmentation")
        resized = cv2.resize(model_img, (384, 512))
        cv2.imwrite(str(resized_path), resized)
        print(f"   Created resized_img.jpg")
    
    # Run Graphonomy
    graphonomy_dir = config.BASE_DIR / "Graphonomy-master"
    graphonomy_script = graphonomy_dir / "exp" / "inference" / "inference.py"
    inference_model = graphonomy_dir / "inference.pth"
    output_path = session_dir / "resized_segmentation_img.png"
    
    print(f"   Graphonomy dir: {graphonomy_dir}")
    print(f"   Script: {graphonomy_script}")
    print(f"   Model: {inference_model}")
    print(f"   Model exists: {inference_model.exists()}")
    
    command = [
        sys.executable,
        str(graphonomy_script),
        "--loadmodel", str(inference_model),
        "--img_path", str(resized_path),
        "--output_path", str(session_dir),
        "--output_name", "resized_segmentation_img",
    ]
    print(f"   Running Graphonomy inference")
    run_subprocess(command, cwd=graphonomy_dir)
    
    print(f"   Checking output: {output_path}")
    print(f"   Output exists: {output_path.exists()}")
    
    if not output_path.exists():
        raise ValueError(f"Graphonomy failed to generate output at: {output_path}")
    
    # Read the original model image (768x1024) for mask processing
    model_path = session_dir / "model.jpg"
    print(f"   Reading model for mask: {model_path}")
    model_img = cv2.imread(str(model_path))
    if model_img is None:
        raise ValueError(f"Cannot read model image from: {model_path}")
    
    print(f"   Model shape: {model_img.shape}")
    
    # Xử lý segmentation mask
    mask_img = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Cannot read segmentation output from: {output_path}")
    
    print(f"   Mask shape before resize: {mask_img.shape}")
    mask_img = cv2.resize(mask_img, (768, 1024))
    print(f"   Mask shape after resize: {mask_img.shape}")
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_img = cv2.erode(mask_img, k)
    
    # Lưu mask để dùng sau
    mask_path = session_dir / "segmentation_mask.png"
    cv2.imwrite(str(mask_path), mask_img)
    
    # Remove background - matching main.py logic exactly
    img_seg = cv2.bitwise_and(model_img, model_img, mask=mask_img)
    back_ground = model_img - img_seg
    img_seg = np.where(img_seg == 0, 215, img_seg)
    
    # Lưu background để dùng sau
    bg_path = session_dir / "background.png"
    cv2.imwrite(str(bg_path), back_ground)
    
    # Save seg_img.png (matching main.py)
    seg_img_path = session_dir / "seg_img.png"
    cv2.imwrite(str(seg_img_path), img_seg)
    
    # Resize and save to HR-VITON directory (already 768x1024, but main.py does this)
    img = cv2.resize(img_seg, (768, 1024))
    seg_path = session_dir / "HR-VITON" / "test" / "test" / "image" / "00001_00.jpg"
    cv2.imwrite(str(seg_path), img)
    
    # Convert segmentation colors to grayscale labels
    parse_v3_dir = session_dir / "HR-VITON" / "test" / "test" / "image-parse-v3"
    generate_image_parse_v3(output_path, parse_v3_dir)
    
    print("✅ Semantic segmentation completed")
    return mask_img, back_ground


def process_densepose(session_dir: Path):
    """Xử lý DensePose"""
    print(f"🦴 Processing DensePose for session: {session_dir.name}")
    
    model_path = session_dir / "model.jpg"  # Use resized model image
    output_pkl = session_dir / "output.pkl"
    output_json = session_dir / "data.json"
    densepose_output_dir = session_dir / "HR-VITON" / "test" / "test" / "image-densepose"
    
    print(f"   Model path: {model_path}")
    print(f"   Model exists: {model_path.exists()}")
    print(f"   Output pkl: {output_pkl}")
    print(f"   Output json: {output_json}")
    print(f"   Output dir: {densepose_output_dir}")
    
    if not model_path.exists():
        raise ValueError(f"Model image not found at: {model_path}")
    
    detectron2_dir = config.BASE_DIR / "detectron2"
    apply_net_script = detectron2_dir / "projects" / "DensePose" / "apply_net.py"
    densepose_config = detectron2_dir / "projects" / "DensePose" / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
    
    print(f"   Detectron2 dir: {detectron2_dir}")
    print(f"   Apply net script: {apply_net_script}")
    print(f"   Script exists: {apply_net_script.exists()}")
    print(f"   Config: {densepose_config}")
    print(f"   Config exists: {densepose_config.exists()}")
    
    # Run DensePose
    command = [
        sys.executable,
        str(apply_net_script),
        "dump",
        str(densepose_config),
        "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
        str(model_path),
        "--output", str(output_pkl),
        "-v",
    ]
    print("   Running DensePose apply_net...")
    run_subprocess(command, cwd=session_dir)
    
    # Check if data.json was created
    if not output_json.exists():
        raise ValueError(f"Failed to generate data.json from DensePose at: {output_json}")
    
    print(f"   JSON created: {output_json}")
    
    # Generate DensePose visualization
    generate_densepose_outputs(session_dir)
    
    print("✅ DensePose completed")


def process_parse_agnostic(session_dir: Path):
    """Xử lý parse agnostic"""
    print(f"📐 Processing parse agnostic for session: {session_dir.name}")
    generate_parse_agnostic_outputs(session_dir)
    print("✅ Parse agnostic completed")


def process_human_agnostic(session_dir: Path):
    """Xử lý human agnostic"""
    print(f"🧍 Processing human agnostic for session: {session_dir.name}")
    generate_human_agnostic_outputs(session_dir)
    print("✅ Human agnostic completed")


def process_hrviton(session_dir: Path):
    """Chạy HR-VITON để tạo ảnh kết quả"""
    print(f"🎯 Running HR-VITON for session: {session_dir.name}")
    
    hrviton_dir = config.BASE_DIR / "HR-VITON"
    dataroot = session_dir / "HR-VITON" / "test"
    session_output_dir = session_dir / "HR-VITON" / "Output"
    session_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear previous outputs for this session
    for png_file in session_output_dir.glob("*.png"):
        png_file.unlink()
    
    # Tạo file test list
    test_list = dataroot / "test_pairs.txt"
    with open(test_list, 'w') as f:
        f.write("00001_00.jpg 00001_00.jpg\n")
    
    # Run HR-VITON with absolute paths
    command = [
        sys.executable,
        "test_generator.py",
        "--cuda", "True",
        "--test_name", session_dir.name,
        "--tocg_checkpoint", "mtviton.pth",
        "--gpu_ids", "0",
        "--gen_checkpoint", "gen.pth",
        "--datasetting", "unpaired",
        "--data_list", "test_pairs.txt",
        "--dataroot", str(dataroot),
        "--output_dir", str(session_output_dir),
    ]
    run_subprocess(command, cwd=hrviton_dir)
    print("✅ HR-VITON completed")


def add_background(result_img_path: Path, mask_img: np.ndarray, background: np.ndarray, add_bg: bool):
    """Thêm background vào ảnh kết quả"""
    img = cv2.imread(str(result_img_path))
    
    if add_bg:
        img = cv2.bitwise_and(img, img, mask=mask_img)
        img = img + background
    
    return img


def full_pipeline(session_dir: Path, add_background_flag: bool = True, progress_cb=None):
    """Pipeline xử lý đầy đủ - Following exact logic from main.py"""
    try:
        # 1. Resize model image to (768, 1024)
        _safe_progress_emit(progress_cb, "start", {"session_id": session_dir.name})
        print(f"📏 Step 1: Resizing model image...")
        model_path = session_dir / "input" / "model.jpg"
        print(f"   Reading from: {model_path}")
        print(f"   File exists: {model_path.exists()}")
        
        if not model_path.exists():
            raise ValueError(f"Model image not found at: {model_path}")
        
        model_img = cv2.imread(str(model_path))
        if model_img is None:
            raise ValueError(f"Cannot read model image from: {model_path}")
        
        print(f"   Original shape: {model_img.shape}")
            
        if model_img.shape[:2] != (1024, 768):
            model_img = cv2.resize(model_img, (768, 1024))
            print(f"   Resized to (768, 1024)")
        else:
            print(f"   Already (768, 1024)")
            
        model_resized_path = session_dir / "model.jpg"
        cv2.imwrite(str(model_resized_path), model_img)
        print(f"   Saved to: {model_resized_path}")
        
        # 1b. Create resized_img.jpg (384x512) for Graphonomy - IMPORTANT: Main.py does this early
        print(f"📏 Step 1b: Creating resized_img.jpg (384x512)...")
        print(f"   Reading from: {model_resized_path}")
        
        img_384_512 = cv2.imread(str(model_resized_path))
        if img_384_512 is None:
            raise ValueError(f"Cannot read resized model image from: {model_resized_path}")
        
        print(f"   Shape before resize: {img_384_512.shape}")
        img_384_512 = cv2.resize(img_384_512, (384, 512))
        print(f"   Shape after resize: {img_384_512.shape}")
        
        resized_img_path = session_dir / "resized_img.jpg"
        cv2.imwrite(str(resized_img_path), img_384_512)
        print(f"   Saved to: {resized_img_path}")
        
        # 2. Process cloth segmentation
        print(f"\n{'='*60}")
        _safe_progress_emit(progress_cb, "cloth_segmentation:start", None)
        process_cloth_mask(session_dir)
        _safe_progress_emit(progress_cb, "cloth_segmentation:done", None)
        
        # 3. Process OpenPose (thay thế OpenPose C++)
        print(f"\n{'='*60}")
        _safe_progress_emit(progress_cb, "openpose:start", None)
        process_openpose(model_resized_path, session_dir)
        _safe_progress_emit(progress_cb, "openpose:done", None)
        
        # 4. Process semantic segmentation (includes grayscale conversion)
        print(f"\n{'='*60}")
        _safe_progress_emit(progress_cb, "segmentation:start", None)
        mask_img, back_ground = process_segmentation(session_dir)
        _safe_progress_emit(progress_cb, "segmentation:done", None)
        
        # 5. Process DensePose
        print(f"\n{'='*60}")
        _safe_progress_emit(progress_cb, "densepose:start", None)
        process_densepose(session_dir)
        _safe_progress_emit(progress_cb, "densepose:done", None)
        
        # 6. Process parse agnostic
        print(f"\n{'='*60}")
        _safe_progress_emit(progress_cb, "parse_agnostic:start", None)
        process_parse_agnostic(session_dir)
        _safe_progress_emit(progress_cb, "parse_agnostic:done", None)
        
        # 7. Process human agnostic
        print(f"\n{'='*60}")
        _safe_progress_emit(progress_cb, "human_agnostic:start", None)
        process_human_agnostic(session_dir)
        _safe_progress_emit(progress_cb, "human_agnostic:done", None)
        
        # 8. Run HR-VITON
        print(f"\n{'='*60}")
        _safe_progress_emit(progress_cb, "hrviton:start", None)
        process_hrviton(session_dir)
        _safe_progress_emit(progress_cb, "hrviton:done", None)
        
        # 9. Find result image(s) - Process ALL results like main.py
        print(f"\n{'='*60}")
        print("🖼️ Step 9: Processing final result(s)...")
        output_dir = session_dir / "HR-VITON" / "Output"
        result_files = sorted(output_dir.glob("*.png"))
        
        if not result_files:
            raise ValueError("No result image generated")
        
        print(f"   Found {len(result_files)} result file(s)")
        
        # 10. Process all result files (like main.py's loop through glob results)
        final_output_paths = []
        for idx, result_img_path in enumerate(result_files):
            print(f"   Processing result {idx+1}/{len(result_files)}: {result_img_path.name}")
            
            # Add background if needed
            final_img = add_background(result_img_path, mask_img, back_ground, add_background_flag)
            
            # Save to session output
            if len(result_files) == 1:
                final_output_path = session_dir / "output" / "final_result.png"
            else:
                final_output_path = session_dir / "output" / f"final_result_{idx}.png"
            
            cv2.imwrite(str(final_output_path), final_img)
            final_output_paths.append(final_output_path)
            
            # Also update the original file in HR-VITON/Output (matching main.py behavior)
            cv2.imwrite(str(result_img_path), final_img)
        
        print(f"\n{'='*60}")
        print(f"🎉 Processing completed for session: {session_dir.name}")
        print(f"   Generated {len(final_output_paths)} final image(s)")
        
        # Return first result path (or list if multiple)
        _safe_progress_emit(progress_cb, "completed", {
            "result_url": build_content_url(final_output_paths[0] if len(final_output_paths) >= 1 else (session_dir / "output" / "final_result.png")),
            "num_results": len(final_output_paths)
        })
        return final_output_paths[0] if len(final_output_paths) == 1 else final_output_paths
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ Error in pipeline: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*60}")
        raise


# =========================
# API Endpoints
# =========================
@app.get("/", tags=["Health"], summary="API root metadata")
async def root():
    return {
        "message": "Virtual Try-On API",
        "version": "1.0.0",
        "endpoints": {
            "upload_images": "/upload",
            "process": "/process/{session_id}",
            "result": "/result/{session_id}",
            "health": "/health"
        }
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    response_model=HealthResponse,
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "openpose_model_loaded": True,
                    }
                }
            },
        }
    },
)
async def health_check():
    return HealthResponse(
        status="healthy",
        openpose_model_loaded=body_estimation is not None,
    )


@app.post(
    "/upload",
    tags=["Upload"],
    summary="Upload model and cloth images",
    response_model=UploadResponse,
    responses={
        200: {
            "description": "Images uploaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "f1e2d3c4-5678-90ab-cdef-1234567890ab",
                        "status": "uploaded",
                        "message": "Images uploaded successfully. Use /process/{session_id} to start processing.",
                    }
                }
            },
        }
    },
)
async def upload_images(
    model_image: UploadFile = File(..., description="Ảnh người mẫu"),
    cloth_image: UploadFile = File(..., description="Ảnh quần áo")
):
    """
    Upload ảnh người mẫu và quần áo
    Returns: session_id để track request
    """
    # Tạo session ID unique
    session_id = str(uuid.uuid4())
    session_dir = create_session_dir(session_id)
    
    try:
        # Lưu model image
        model_path = session_dir / "input" / "model.jpg"
        with open(model_path, "wb") as f:
            content = await model_image.read()
            f.write(content)
        
        # Lưu cloth image
        cloth_path = session_dir / "input" / "cloth.jpg"
        with open(cloth_path, "wb") as f:
            content = await cloth_image.read()
            f.write(content)
        
        return UploadResponse(
            session_id=session_id,
            status="uploaded",
            message="Images uploaded successfully. Use /process/{session_id} to start processing.",
        )
        
    except Exception as e:
        # Cleanup nếu có lỗi
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post(
    "/process/{session_id}",
    tags=["Process"],
    summary="Run virtual try-on pipeline",
    response_model=TryOnResponse,
    responses={
        200: {
            "description": "Processing completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "f1e2d3c4-5678-90ab-cdef-1234567890ab",
                        "status": "completed",
                        "message": "Processing completed successfully",
                        "result_image_url": "/content/Fashion-U-Want-Virtual-Try-On/temp_sessions/f1e2d3c4-5678-90ab-cdef-1234567890ab/output/final_result.png",
                    }
                }
            },
        }
    },
)
async def process_tryon(
    session_id: str,
    background_tasks: BackgroundTasks,
    add_background: bool = True
):
    """
    Xử lý virtual try-on
    """
    session_dir = config.TEMP_DIR / session_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if images exist
    model_path = session_dir / "input" / "model.jpg"
    cloth_path = session_dir / "input" / "cloth.jpg"
    
    if not model_path.exists() or not cloth_path.exists():
        raise HTTPException(status_code=400, detail="Images not found. Please upload first.")
    
    try:
        # Chạy pipeline trong thread pool để không block
        loop = asyncio.get_event_loop()

        # Prepare progress queue for realtime updates
        if session_id not in PROGRESS_QUEUES:
            PROGRESS_QUEUES[session_id] = asyncio.Queue()

        def progress_cb(payload: dict):
            # Called from worker thread; forward to websocket consumers via queue
            asyncio.run_coroutine_threadsafe(PROGRESS_QUEUES[session_id].put(payload), loop)

        result_path = await loop.run_in_executor(
            executor,
            full_pipeline,
            session_dir,
            add_background,
            progress_cb
        )
        
        # Normalize to a single Path for URL building
        result_file = result_path[0] if isinstance(result_path, list) and len(result_path) > 0 else result_path
        content_url = build_content_url(result_file if isinstance(result_file, Path) else (session_dir / "output" / "final_result.png"))

        return TryOnResponse(
            session_id=session_id,
            status="completed",
            message="Processing completed successfully",
            result_image_url=content_url,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.websocket("/ws/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    await websocket.accept()
    # Ensure queue exists
    if session_id not in PROGRESS_QUEUES:
        PROGRESS_QUEUES[session_id] = asyncio.Queue()

    queue = PROGRESS_QUEUES[session_id]
    try:
        await websocket.send_json({"event": "connected", "session_id": session_id})
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        return


@app.get(
    "/result/{session_id}",
    tags=["Result"],
    summary="Get result image by session",
    responses={
        200: {
            "description": "PNG image content",
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
        },
        404: {"description": "Result not found or processing not completed"},
    },
)
async def get_result(session_id: str):
    """
    Lấy ảnh kết quả
    """
    session_dir = config.TEMP_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Prefer the canonical final_result.png, but be resilient
    candidates = [
        session_dir / "output" / "final_result.png",
    ]

    # Include any numbered results
    candidates.extend(sorted((session_dir / "output").glob("final_result_*.png")))

    # Fallback to HR-VITON outputs if output/ is empty for some reason
    candidates.extend(sorted((session_dir / "HR-VITON" / "Output").glob("*.png")))

    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Result not found or processing not completed")

    # Read image file as bytes and return with explicit image headers
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    
    # Return Response with explicit binary content and headers
    from fastapi.responses import Response
    
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            # CORS headers
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            # Force browser to treat as image
            "Content-Type": "image/png",
            "Content-Disposition": f'inline; filename="{file_path.name}"',
            # Cache control
            "Cache-Control": "public, max-age=3600",
            # Prevent ngrok from injecting HTML
            "X-Content-Type-Options": "nosniff",
            # Additional headers for binary content
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(image_bytes))
        }
    )


@app.delete(
    "/session/{session_id}",
    tags=["Sessions"],
    summary="Delete session and cleanup files",
    response_model=DeleteSessionResponse,
)
async def delete_session(session_id: str):
    """
    Xóa session và dọn dẹp file
    """
    session_dir = config.TEMP_DIR / session_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        shutil.rmtree(session_dir)
        return DeleteSessionResponse(
            session_id=session_id,
            status="deleted",
            message="Session cleaned up successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")



# =========================
# Main
# =========================
if __name__ == "__main__":
    # Chạy server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # Dùng 1 worker vì đã có ThreadPoolExecutor
    )
