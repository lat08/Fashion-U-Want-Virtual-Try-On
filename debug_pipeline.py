"""
Debug Script for Virtual Try-On Pipeline
Chạy pipeline hoàn chỉnh mà không cần API server
"""

import os
import sys
import uuid
import shutil
import cv2
import json
import copy
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

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
    TEMP_DIR = BASE_DIR / "debug_sessions"
    
    # Debug settings
    SAVE_INTERMEDIATE = True  # Lưu tất cả ảnh trung gian
    VERBOSE = True  # In chi tiết từng bước


config = Config()
config.TEMP_DIR.mkdir(exist_ok=True)


# =========================
# Constants & Helpers
# =========================
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


class DebugLogger:
    """Logger với màu sắc và timestamp"""
    
    @staticmethod
    def step(step_num, message):
        print(f"\n{'='*80}")
        print(f"📍 STEP {step_num}: {message}")
        print(f"{'='*80}")
    
    @staticmethod
    def info(message):
        print(f"ℹ️  {message}")
    
    @staticmethod
    def success(message):
        print(f"✅ {message}")
    
    @staticmethod
    def warning(message):
        print(f"⚠️  {message}")
    
    @staticmethod
    def error(message):
        print(f"❌ {message}")
    
    @staticmethod
    def detail(key, value):
        print(f"   • {key}: {value}")


logger = DebugLogger()


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
        if config.VERBOSE and result.stdout:
            logger.detail("stdout", result.stdout[:200])
        return result
    except subprocess.CalledProcessError as exc:
        logger.error(f"Command failed with exit code {exc.returncode}")
        if exc.stderr:
            logger.error(f"stderr: {exc.stderr[:500]}")
        raise


def create_session_dir(session_id: str) -> Path:
    """Tạo thư mục riêng cho mỗi session"""
    session_dir = config.TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Tạo các thư mục con cần thiết
    dirs = [
        "input",
        "output",
        "HR-VITON/test/test/image",
        "HR-VITON/test/test/cloth",
        "HR-VITON/test/test/cloth-mask",
        "HR-VITON/test/test/openpose_img",
        "HR-VITON/test/test/openpose_json",
        "HR-VITON/test/test/image-parse-v3",
        "HR-VITON/test/test/image-parse-agnostic-v3.2",
        "HR-VITON/test/test/agnostic-v3.2",
        "HR-VITON/test/test/image-densepose",
        "HR-VITON/Output",
    ]
    
    for d in dirs:
        (session_dir / d).mkdir(parents=True, exist_ok=True)
    
    logger.success(f"Created session directory: {session_dir}")
    return session_dir


def generate_image_parse_v3(segmentation_path: Path, output_dir: Path) -> Path:
    """Convert Graphonomy RGB segmentation map to grayscale label map."""
    logger.info("Converting RGB segmentation to grayscale labels...")
    
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
    
    logger.success(f"Generated parse-v3: {output_path}")
    return output_path


def generate_densepose_outputs(session_dir: Path) -> Path:
    """Generate DensePose visualization matching get_densepose.py behavior."""
    logger.info("Generating DensePose visualization...")
    
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
    
    logger.success(f"Generated DensePose: {output_path}")
    return output_path


def generate_parse_agnostic_outputs(session_dir: Path):
    """Recreate get_parse_agnostic.py logic using session-scoped paths."""
    logger.info("Generating parse agnostic masks...")
    
    data_path = session_dir / "HR-VITON" / "test" / "test"
    image_dir = data_path / "image"
    parse_dir = data_path / "image-parse-v3"
    openpose_dir = data_path / "openpose_json"
    output_dir = data_path / "image-parse-agnostic-v3.2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists() or not parse_dir.exists() or not openpose_dir.exists():
        logger.warning("Required directories missing for parse agnostic")
        return

    count = 0
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
        count += 1
    
    logger.success(f"Generated {count} parse agnostic mask(s)")


def generate_human_agnostic_outputs(session_dir: Path):
    """Recreate get_agnostic2.py logic using session-scoped paths."""
    logger.info("Generating human agnostic images...")
    
    data_path = session_dir / "HR-VITON" / "test" / "test"
    image_dir = data_path / "image"
    parse_dir = data_path / "image-parse-v3"
    openpose_dir = data_path / "openpose_json"
    output_dir = data_path / "agnostic-v3.2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists() or not parse_dir.exists() or not openpose_dir.exists():
        logger.warning("Required directories missing for human agnostic")
        return

    count = 0
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
        count += 1
    
    logger.success(f"Generated {count} human agnostic image(s)")


# =========================
# Pipeline Steps
# =========================

def step1_resize_model(session_dir: Path, model_input_path: Path) -> Path:
    """Step 1: Resize model image to (768, 1024)"""
    logger.step(1, "Resize model image to (768, 1024)")
    
    model_img = cv2.imread(str(model_input_path))
    if model_img is None:
        raise ValueError(f"Cannot read model image from: {model_input_path}")
    
    logger.detail("Original shape", model_img.shape)
    
    if model_img.shape[:2] != (1024, 768):
        model_img = cv2.resize(model_img, (768, 1024))
        logger.info("Resized to (768, 1024)")
    else:
        logger.info("Already correct size")
    
    model_resized_path = session_dir / "model.jpg"
    cv2.imwrite(str(model_resized_path), model_img)
    
    logger.success(f"Saved resized model: {model_resized_path}")
    return model_resized_path


def step2_create_graphonomy_input(session_dir: Path, model_resized_path: Path) -> Path:
    """Step 2: Create 384x512 image for Graphonomy"""
    logger.step(2, "Create resized_img.jpg (384x512) for Graphonomy")
    
    img = cv2.imread(str(model_resized_path))
    if img is None:
        raise ValueError(f"Cannot read model image")
    
    img_384_512 = cv2.resize(img, (384, 512))
    resized_img_path = session_dir / "resized_img.jpg"
    cv2.imwrite(str(resized_img_path), img_384_512)
    
    logger.detail("Shape", img_384_512.shape)
    logger.success(f"Saved: {resized_img_path}")
    return resized_img_path


def step3_cloth_segmentation(session_dir: Path):
    """Step 3: Process cloth segmentation"""
    logger.step(3, "Cloth Segmentation")
    
    input_path = str(session_dir / "input" / "cloth.jpg")
    cloth_output = str(session_dir / "HR-VITON" / "test" / "test" / "cloth")
    mask_output = str(session_dir / "HR-VITON" / "test" / "test" / "cloth-mask")
    
    if not Path(input_path).exists():
        raise ValueError(f"Cloth image not found at: {input_path}")
    
    logger.info(f"Processing: {input_path}")
    process_cloth_segmentation(
        input_path=input_path,
        cloth_output_path=cloth_output,
        mask_output_path=mask_output
    )
    
    logger.success("Cloth segmentation completed")


def step4_openpose(model_resized_path: Path, session_dir: Path, body_estimation):
    """Step 4: OpenPose detection"""
    logger.step(4, "OpenPose Keypoint Detection")
    
    oriImg = cv2.imread(str(model_resized_path))
    if oriImg is None:
        raise ValueError(f"Cannot read image: {model_resized_path}")
    
    logger.detail("Image shape", oriImg.shape)
    
    # Detect pose
    candidate, subset = body_estimation(oriImg)
    
    if len(subset) == 0:
        raise ValueError("No person detected in image")
    
    logger.detail("Persons detected", len(subset))
    
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
    
    logger.success(f"OpenPose completed: {len(result['people'])} person(s) detected")
    logger.detail("JSON saved", output_json_path)
    logger.detail("Image saved", output_img_path)


def step5_graphonomy(session_dir: Path, resized_img_path: Path):
    """Step 5: Semantic segmentation with Graphonomy"""
    logger.step(5, "Graphonomy Semantic Segmentation")
    
    graphonomy_dir = config.BASE_DIR / "Graphonomy-master"
    graphonomy_script = graphonomy_dir / "exp" / "inference" / "inference.py"
    inference_model = graphonomy_dir / "inference.pth"
    output_path = session_dir / "resized_segmentation_img.png"
    
    logger.detail("Graphonomy dir", graphonomy_dir)
    logger.detail("Model exists", inference_model.exists())
    
    command = [
        sys.executable,
        str(graphonomy_script),
        "--loadmodel", str(inference_model),
        "--img_path", str(resized_img_path),
        "--output_path", str(session_dir),
        "--output_name", "resized_segmentation_img",
    ]
    
    logger.info("Running Graphonomy inference...")
    run_subprocess(command, cwd=graphonomy_dir)
    
    if not output_path.exists():
        raise ValueError(f"Graphonomy failed to generate output at: {output_path}")
    
    logger.success(f"Segmentation generated: {output_path}")
    return output_path


def step6_process_segmentation_mask(session_dir: Path, segmentation_path: Path):
    """Step 6: Process segmentation mask and remove background"""
    logger.step(6, "Process Segmentation Mask")
    
    model_path = session_dir / "model.jpg"
    model_img = cv2.imread(str(model_path))
    if model_img is None:
        raise ValueError(f"Cannot read model image from: {model_path}")
    
    logger.detail("Model shape", model_img.shape)
    
    # Xử lý segmentation mask
    mask_img = cv2.imread(str(segmentation_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Cannot read segmentation output from: {segmentation_path}")
    
    mask_img = cv2.resize(mask_img, (768, 1024))
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
    
    # Save seg_img.png
    seg_img_path = session_dir / "seg_img.png"
    cv2.imwrite(str(seg_img_path), img_seg)
    
    # Resize and save to HR-VITON directory
    img = cv2.resize(img_seg, (768, 1024))
    seg_path = session_dir / "HR-VITON" / "test" / "test" / "image" / "00001_00.jpg"
    cv2.imwrite(str(seg_path), img)
    
    # Convert segmentation colors to grayscale labels
    parse_v3_dir = session_dir / "HR-VITON" / "test" / "test" / "image-parse-v3"
    generate_image_parse_v3(segmentation_path, parse_v3_dir)
    
    logger.success("Segmentation mask processed")
    logger.detail("Mask saved", mask_path)
    logger.detail("Background saved", bg_path)
    logger.detail("Seg image saved", seg_path)
    
    return mask_img, back_ground


def step7_densepose(session_dir: Path):
    """Step 7: DensePose detection"""
    logger.step(7, "DensePose Detection")
    
    model_path = session_dir / "model.jpg"
    output_pkl = session_dir / "output.pkl"
    output_json = session_dir / "data.json"
    
    if not model_path.exists():
        raise ValueError(f"Model image not found at: {model_path}")
    
    detectron2_dir = config.BASE_DIR / "detectron2"
    apply_net_script = detectron2_dir / "projects" / "DensePose" / "apply_net.py"
    densepose_config = detectron2_dir / "projects" / "DensePose" / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
    
    logger.detail("Detectron2 dir", detectron2_dir)
    logger.detail("Config exists", densepose_config.exists())
    
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
    
    logger.info("Running DensePose apply_net...")
    run_subprocess(command, cwd=session_dir)
    
    # Check if data.json was created
    if not output_json.exists():
        raise ValueError(f"Failed to generate data.json from DensePose at: {output_json}")
    
    logger.detail("JSON created", output_json)
    
    # Generate DensePose visualization
    generate_densepose_outputs(session_dir)
    
    logger.success("DensePose completed")


def step8_parse_agnostic(session_dir: Path):
    """Step 8: Generate parse agnostic masks"""
    logger.step(8, "Generate Parse Agnostic Masks")
    generate_parse_agnostic_outputs(session_dir)


def step9_human_agnostic(session_dir: Path):
    """Step 9: Generate human agnostic images"""
    logger.step(9, "Generate Human Agnostic Images")
    generate_human_agnostic_outputs(session_dir)


def step10_hrviton(session_dir: Path):
    """Step 10: Run HR-VITON"""
    logger.step(10, "Run HR-VITON")
    
    hrviton_dir = config.BASE_DIR / "HR-VITON"
    dataroot = session_dir / "HR-VITON" / "test"
    session_output_dir = session_dir / "HR-VITON" / "Output"
    session_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear previous outputs
    for png_file in session_output_dir.glob("*.png"):
        png_file.unlink()
    
    # Tạo file test list
    test_list = dataroot / "test_pairs.txt"
    with open(test_list, 'w') as f:
        f.write("00001_00.jpg 00001_00.jpg\n")
    
    logger.detail("Data root", dataroot)
    logger.detail("Output dir", session_output_dir)
    
    # Run HR-VITON
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
    
    logger.info("Running HR-VITON test_generator...")
    run_subprocess(command, cwd=hrviton_dir)
    
    logger.success("HR-VITON completed")


def step11_finalize(session_dir: Path, mask_img: np.ndarray, back_ground: np.ndarray, add_background_flag: bool):
    """Step 11: Process final results"""
    logger.step(11, "Finalize Results")
    
    output_dir = session_dir / "HR-VITON" / "Output"
    result_files = sorted(output_dir.glob("*.png"))
    
    if not result_files:
        raise ValueError("No result image generated")
    
    logger.detail("Found result files", len(result_files))
    
    final_output_paths = []
    for idx, result_img_path in enumerate(result_files):
        logger.info(f"Processing result {idx+1}/{len(result_files)}: {result_img_path.name}")
        
        img = cv2.imread(str(result_img_path))
        
        if add_background_flag:
            img = cv2.bitwise_and(img, img, mask=mask_img)
            img = img + back_ground
        
        # Save to session output
        if len(result_files) == 1:
            final_output_path = session_dir / "output" / "final_result.png"
        else:
            final_output_path = session_dir / "output" / f"final_result_{idx}.png"
        
        cv2.imwrite(str(final_output_path), img)
        final_output_paths.append(final_output_path)
        
        # Also update the original file
        cv2.imwrite(str(result_img_path), img)
    
    logger.success(f"Generated {len(final_output_paths)} final image(s)")
    for path in final_output_paths:
        logger.detail("Final output", path)
    
    return final_output_paths


# =========================
# Main Debug Function
# =========================

def debug_full_pipeline(model_image_path: str, cloth_image_path: str, add_background: bool = True):
    """
    Chạy toàn bộ pipeline với debug đầy đủ
    
    Args:
        model_image_path: Đường dẫn đến ảnh người mẫu
        cloth_image_path: Đường dẫn đến ảnh quần áo
        add_background: Có thêm background vào kết quả hay không
    """
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("🚀 VIRTUAL TRY-ON DEBUG PIPELINE")
    print("="*80)
    
    # Validate input files
    model_path = Path(model_image_path)
    cloth_path = Path(cloth_image_path)
    
    if not model_path.exists():
        logger.error(f"Model image not found: {model_path}")
        return
    
    if not cloth_path.exists():
        logger.error(f"Cloth image not found: {cloth_path}")
        return
    
    logger.info(f"Model image: {model_path}")
    logger.info(f"Cloth image: {cloth_path}")
    logger.info(f"Add background: {add_background}")
    
    # Create session
    session_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = create_session_dir(session_id)
    
    try:
        # Copy input files
        logger.info("Copying input files...")
        shutil.copy(model_path, session_dir / "input" / "model.jpg")
        shutil.copy(cloth_path, session_dir / "input" / "cloth.jpg")
        
        # Load OpenPose model
        logger.info("Loading OpenPose model...")
        body_estimation = Body(str(config.OPENPOSE_MODEL_PATH), 'body25')
        logger.success("OpenPose model loaded")
        
        # Run pipeline steps
        model_resized_path = step1_resize_model(session_dir, session_dir / "input" / "model.jpg")
        resized_img_path = step2_create_graphonomy_input(session_dir, model_resized_path)
        step3_cloth_segmentation(session_dir)
        step4_openpose(model_resized_path, session_dir, body_estimation)
        segmentation_path = step5_graphonomy(session_dir, resized_img_path)
        mask_img, back_ground = step6_process_segmentation_mask(session_dir, segmentation_path)
        step7_densepose(session_dir)
        step8_parse_agnostic(session_dir)
        step9_human_agnostic(session_dir)
        step10_hrviton(session_dir)
        final_paths = step11_finalize(session_dir, mask_img, back_ground, add_background)
        
        # Summary
        elapsed = datetime.now() - start_time
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        logger.detail("Session ID", session_id)
        logger.detail("Session directory", session_dir)
        logger.detail("Total time", f"{elapsed.total_seconds():.2f}s")
        logger.detail("Final outputs", len(final_paths))
        
        for i, path in enumerate(final_paths, 1):
            print(f"\n📸 Result {i}: {path}")
        
        print("\n" + "="*80)
        
        return session_dir, final_paths
        
    except Exception as e:
        elapsed = datetime.now() - start_time
        print("\n" + "="*80)
        print("❌ PIPELINE FAILED!")
        print("="*80)
        logger.error(f"Error: {str(e)}")
        logger.detail("Error type", type(e).__name__)
        logger.detail("Time before failure", f"{elapsed.total_seconds():.2f}s")
        
        import traceback
        print("\n📋 Full traceback:")
        print("-"*80)
        traceback.print_exc()
        print("-"*80)
        
        return None, None


# =========================
# CLI Interface
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Virtual Try-On Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to model image")
    parser.add_argument("--cloth", type=str, required=True, help="Path to cloth image")
    parser.add_argument("--background", type=bool, default=True, help="Add background to result")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config.VERBOSE = args.verbose
    
    debug_full_pipeline(
        model_image_path=args.model,
        cloth_image_path=args.cloth,
        add_background=args.background
    )
