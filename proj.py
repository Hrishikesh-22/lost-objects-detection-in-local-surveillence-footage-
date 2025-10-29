# app/main.py
"""
FastAPI app for Lost Objects Detection (CPU-friendly).
Endpoints:
  - GET  /health          : basic health check
  - POST /predict        : upload an image or video file (multipart/form-data "file")
Notes:
  - Replace `load_model`, `preprocess_image`, `run_inference`, `postprocess` with your model-specific code.
  - This implementation attempts to handle images first, then videos.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import torch
import time
import logging

# Config (edit as needed)
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.pt")
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 50 * 1024 * 1024))  # 50 MB default
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 5))  # process every 5th frame for videos
DEVICE = torch.device("cpu")

app = FastAPI(title="Lost Objects Detection API")
logger = logging.getLogger("uvicorn.error")


# -------------------------
# Model helpers - EDIT HERE
# -------------------------
def load_model(path: str = MODEL_PATH) -> Any:
    """
    Load your model and return a model object with a `.eval()` method.
    Replace the body of this function with code that loads your trained weights.
    Examples:
      - For a simple torch.save(torch_model.state_dict()) you may need to build model architecture then load_state_dict
      - If you saved the whole model with torch.save(model), torch.load will work directly (less recommended)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    # Example: load a full model object (works if you saved entire model with torch.save(model))
    try:
        model = torch.load(path, map_location=DEVICE)
        model.eval()
        return model
    except Exception:
        # Example alternative: if you saved state_dict, construct model and load
        raise RuntimeError("Model load failed — replace load_model with your model init + load_state_dict logic.")


def preprocess_image(image_bgr: np.ndarray) -> Any:
    """
    Convert OpenCV BGR image to the input expected by your model.
    Return a tensor or whatever your model expects.
    Example steps (PyTorch):
      - convert BGR -> RGB
      - resize / pad to model input size
      - normalize / to tensor / add batch dimension
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Example placeholder: resize to 640x640 and normalize to [0,1]
    img_resized = cv2.resize(img_rgb, (640, 640))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # C,H,W
    tensor = tensor.unsqueeze(0).to(DEVICE)  # 1,C,H,W
    return tensor


def run_inference(model: Any, model_input: Any) -> Any:
    """
    Run model inference.
    Replace this with your model forward pass and return raw model outputs.
    """
    with torch.no_grad():
        out = model(model_input)  # adjust if your model returns tuple or requires different call
    return out


def postprocess(raw_outputs: Any, orig_w: int, orig_h: int) -> List[Dict]:
    """
    Convert raw model outputs to a serializable list of detection dicts:
      [{ "label": "obj", "score": 0.92, "bbox": [x1,y1,x2,y2] }, ...]
    You must replace this with logic that matches your model's output format.
    """
    # Placeholder: no detections
    return []


# -------------------------
# App startup: load model
# -------------------------
model = None


@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        # keep server up even if model fails to load so /health works and we can debug
        logger.exception("Failed to load model at startup: %s", e)
        model = None


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health() -> Dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Accepts multipart/form-data with a single file field named "file".
    Tries to treat file as image first; if not an image, tries as video.
    Returns JSON with detections.
    """
    # quick size check
    file_size = 0
    try:
        contents = await file.read()
        file_size = len(contents)
    finally:
        # reset underlying file-like for further use — but we have contents in memory
        pass

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_UPLOAD_SIZE} bytes")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # Try image path
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        try:
            t0 = time.time()
            input_tensor = preprocess_image(img)
            raw = run_inference(model, input_tensor)
            detections = postprocess(raw, orig_w=img.shape[1], orig_h=img.shape[0])
            latency = time.time() - t0
            return JSONResponse(content={
                "type": "image",
                "detections": detections,
                "meta": {"width": img.shape[1], "height": img.shape[0], "latency_s": latency}
            })
        except Exception as e:
            logger.exception("Image inference error: %s", e)
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Otherwise write bytes to a temp file and open as video
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1] or ".mp4")
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is neither a recognizable image nor a playable video.")

    try:
        frame_idx = 0
        frame_results = []
        t_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SKIP == 0:
                try:
                    input_tensor = preprocess_image(frame)
                    raw = run_inference(model, input_tensor)
                    detections = postprocess(raw, orig_w=frame.shape[1], orig_h=frame.shape[0])
                    frame_results.append({
                        "frame": frame_idx,
                        "detections": detections,
                    })
                except Exception as e:
                    # log and continue — don't kill the whole video on one frame error
                    logger.exception("Inference failed on frame %d: %s", frame_idx, e)
            frame_idx += 1

        total_time = time.time() - t_start
        return JSONResponse(content={
            "type": "video",
            "frames_scanned": frame_idx,
            "frames_processed": len(frame_results),
            "results": frame_results,
            "meta": {"processing_time_s": total_time}
        })
    finally:
        cap.release()
        # schedule deletion of temp file in the background (safer than immediate deletion)
        if background_tasks is not None:
            background_tasks.add_task(lambda p: os.remove(p) if os.path.exists(p) else None, tmp_path)
        else:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# -------------------------
# If run directly (local dev)
# -------------------------
if __name__ == "__main__":
    # For local debugging only. In production we use the Docker CMD to run uvicorn.
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=False)
