import os
from io import BytesIO
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel

# Optional deps: import defensively so missing modules don't crash startup
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore

from PIL import Image, ImageStat

from database import create_document
from schemas import FacemaxAnalysis

app = FastAPI(title="Facemaxxing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    message: str


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Facemaxxing backend running"}


def _euclid(ax: float, ay: float, bx: float, by: float) -> float:
    import math
    return math.hypot(ax - bx, ay - by)


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns a facemaxxing-style analysis.
    Uses Mediapipe Face Mesh if available; otherwise falls back to simple heuristics.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload")

    # Try decoding image
    img_bgr = None  # OpenCV BGR format if available
    pil_img: Optional[Image.Image] = None

    if cv2 is not None and np is not None:
        try:
            np_arr = np.frombuffer(content, dtype='uint8')  # type: ignore
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            img_bgr = None

    if img_bgr is None:
        try:
            pil_img = Image.open(BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=415, detail="Unsupported image format")

    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
    else:
        assert pil_img is not None
        w, h = pil_img.size

    metrics: Dict[str, Any] = {"image_width": w, "image_height": h}

    # Attempt landmark analysis only if both mediapipe and numpy are available
    landmarks = None
    if mp is not None and np is not None and img_bgr is not None:
        try:
            mp_face = mp.solutions.face_mesh
            with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                rgb = img_bgr[:, :, ::-1]
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
        except Exception:
            landmarks = None

    # Initialize scores
    jawline_score = 5.0
    cheekbone_score = 5.0
    eye_score = 5.0
    skin_score = 5.0
    symmetry_score = 5.0

    if landmarks is not None and np is not None:
        lm = landmarks.landmark

        def pt(i: int):
            return (lm[i].x * w, lm[i].y * h)

        face_width = _euclid(*pt(127), *pt(356))
        jaw_width = _euclid(*pt(234), *pt(454))
        lower_face = _euclid(*pt(152), *pt(6))
        left_eye_open = _euclid(*pt(159), *pt(145))
        right_eye_open = _euclid(*pt(386), *pt(374))
        if len(lm) > 473:
            interpupil = _euclid(*pt(468), *pt(473))
        else:
            interpupil = _euclid(*pt(133), *pt(362))

        symmetry = 1.0 - min(1.0, abs(left_eye_open - right_eye_open) / (left_eye_open + right_eye_open + 1e-6))
        jaw_ratio = jaw_width / (lower_face + 1e-6)
        cheek_ratio = face_width / (lower_face + 1e-6)
        eye_ratio = (left_eye_open + right_eye_open) / (interpupil + 1e-6)

        metrics.update({
            "face_width": float(face_width),
            "jaw_width": float(jaw_width),
            "lower_face_height": float(lower_face),
            "left_eye_open": float(left_eye_open),
            "right_eye_open": float(right_eye_open),
            "interpupil": float(interpupil),
            "symmetry_proxy": float(symmetry),
            "jaw_ratio": float(jaw_ratio),
            "cheek_ratio": float(cheek_ratio),
            "eye_ratio": float(eye_ratio),
        })

        # Heuristic mapping to 1-10 ranges
        jawline_score = max(1.0, min(10.0, 10.0 - abs(jaw_ratio - 1.2) * 20))
        cheekbone_score = max(1.0, min(10.0, 10.0 - abs(cheek_ratio - 1.7) * 10))
        eye_score = max(1.0, min(10.0, 10.0 - abs(eye_ratio - 0.3) * 60))
        symmetry_score = max(1.0, min(10.0, 1.0 + symmetry * 9.0))
    else:
        # Fallback: simple skin clarity proxy using grayscale stddev via PIL
        assert pil_img is not None or img_bgr is not None
        if pil_img is None:
            # Convert OpenCV BGR to PIL
            from PIL import Image as _Image
            rgb = img_bgr[:, :, ::-1]
            pil_img = _Image.fromarray(rgb)
        gray = pil_img.convert('L')
        stats = ImageStat.Stat(gray)
        # stats.stddev returns stdev per channel, for L it's a single value
        skin_std = float(stats.stddev[0] if isinstance(stats.stddev, (list, tuple)) else stats.stddev)
        metrics.update({"skin_texture_std": skin_std})
        skin_score = max(1.0, min(10.0, 6.0 + (15.0 - min(15.0, skin_std)) * 0.2))
        # Other features default to mid
        jawline_score = 5.0
        cheekbone_score = 5.0
        eye_score = 5.0
        symmetry_score = 5.0

    score = float(round((jawline_score*0.25 + cheekbone_score*0.2 + eye_score*0.2 + skin_score*0.15 + symmetry_score*0.2), 1))

    def qual(s: float) -> str:
        return (
            "excellent" if s >= 8.5 else
            "good" if s >= 7 else
            "average" if s >= 5 else
            "below average" if s >= 3.5 else
            "needs improvement"
        )

    review = {
        "jawline": f"Jawline definition is {qual(jawline_score)}.",
        "cheekbones": f"Cheekbone prominence looks {qual(cheekbone_score)}.",
        "eyes": f"Eye area proportions are {qual(eye_score)}.",
        "skin": f"Skin clarity appears {qual(skin_score)}.",
        "symmetry": f"Facial symmetry is {qual(symmetry_score)}.",
    }

    tips = [
        "Practice proper tongue posture (mewing) consistently.",
        "Keep a simple skincare routine: cleanse, moisturize, SPF daily.",
        "Maintain facial hair and grooming for clean edges.",
        "Choose a haircut that adds structure around the jaw/cheekbones.",
        "Prioritize sleep, hydration, and nutrition for skin and eye area.",
    ]

    doc = FacemaxAnalysis(
        filename=getattr(file, 'filename', None),
        score=score,
        review=review,
        tips=tips,
        metrics=metrics,
        image_width=w,
        image_height=h,
    )

    try:
        create_document('facemaxanalysis', doc)
    except Exception:
        pass

    return JSONResponse(content={
        "score": score,
        "review": review,
        "tips": tips,
        "metrics": metrics,
    })


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
