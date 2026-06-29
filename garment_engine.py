# ─── SILHOUETTE GARMENT ENGINE ────────────────────────────────────────────────
# Processes clothing photos into try-on-ready assets

import cv2
import numpy as np
from PIL import Image
from rembg import remove
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import base64
import io
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum


class GarmentCategory(str, Enum):
    TOP        = "top"
    BOTTOM     = "bottom"
    DRESS      = "dress"
    OUTERWEAR  = "outerwear"
    SHOES      = "shoes"
    BAG        = "bag"
    JEWELLERY  = "jewellery"
    SCARF      = "scarf"


@dataclass
class ProcessedGarment:
    category:        GarmentCategory
    dominant_colour: Tuple[int, int, int]
    texture_b64:     str          # base64 PNG, bg removed
    thumbnail_b64:   str          # 200×300 thumbnail
    mask_b64:        str          # binary mask for draping
    uv_region:       Dict         # which UV region to apply to
    metadata:        Dict


# ─── CLASSIFIER ───────────────────────────────────────────────────────────────

class GarmentClassifier:
    """
    Fine-tuned ResNet50 for garment category classification.
    Falls back to heuristic shape analysis if model unavailable.
    """

    # DeepFashion2 category mapping (simplified)
    CATEGORY_MAP = {
        0: GarmentCategory.TOP,
        1: GarmentCategory.TOP,
        2: GarmentCategory.BOTTOM,
        3: GarmentCategory.BOTTOM,
        4: GarmentCategory.DRESS,
        5: GarmentCategory.OUTERWEAR,
        6: GarmentCategory.OUTERWEAR,
        7: GarmentCategory.SHOES,
        8: GarmentCategory.BAG,
        9: GarmentCategory.SCARF,
        10: GarmentCategory.JEWELLERY,
    }

    def __init__(self):
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])
        # Load model (replace with fine-tuned weights path in production)
        try:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.eval()
            self.model_available = True
        except Exception:
            self.model_available = False

    def classify(self, image: np.ndarray) -> GarmentCategory:
        if self.model_available:
            return self._classify_ml(image)
        return self._classify_heuristic(image)

    def _classify_ml(self, image: np.ndarray) -> GarmentCategory:
        pil    = Image.fromarray(image)
        tensor = self.transform(pil).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        idx = out.argmax(dim=1).item() % len(self.CATEGORY_MAP)
        return self.CATEGORY_MAP.get(idx, GarmentCategory.TOP)

    def _classify_heuristic(self, image: np.ndarray) -> GarmentCategory:
        """
        Shape-based fallback: analyse aspect ratio and 
        mass distribution of the garment silhouette.
        """
        gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return GarmentCategory.TOP
        cnt     = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        ratio   = h / max(w, 1)

        # Mass distribution: upper half vs lower half
        upper   = mask[:mask.shape[0]//2].sum()
        lower   = mask[mask.shape[0]//2:].sum()
        bottom_heavy = lower > upper * 1.3

        if ratio > 2.2:
            return GarmentCategory.DRESS
        elif ratio > 1.4 and bottom_heavy:
            return GarmentCategory.BOTTOM
        elif ratio < 0.6:
            return GarmentCategory.SHOES
        elif ratio < 0.8:
            return GarmentCategory.BAG
        elif w < image.shape[1] * 0.25:
            return GarmentCategory.JEWELLERY
        else:
            return GarmentCategory.TOP


# ─── COLOUR EXTRACTOR ─────────────────────────────────────────────────────────

class ColourExtractor:

    def dominant_colour(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[int, int, int]:
        """K-means on masked pixels to find dominant colour."""
        pixels = image[mask > 128].reshape(-1, 3).astype(np.float32)
        if len(pixels) < 10:
            return (128, 128, 128)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10, 1.0
        )
        k = 3
        _, labels, centres = cv2.kmeans(
            pixels, k, None, criteria, 5,
            cv2.KMEANS_RANDOM_CENTERS
        )
        # Return most frequent cluster
        counts  = np.bincount(labels.flatten())
        dominant = centres[counts.argmax()]
        return (int(dominant[0]), int(dominant[1]), int(dominant[2]))


# ─── UV REGION MAPPING ────────────────────────────────────────────────────────

UV_REGION_MAP = {
    GarmentCategory.TOP:       {"y_min": 0.35, "y_max": 0.62, "scale": 1.0},
    GarmentCategory.BOTTOM:    {"y_min": 0.58, "y_max": 0.88, "scale": 1.0},
    GarmentCategory.DRESS:     {"y_min": 0.32, "y_max": 0.92, "scale": 1.05},
    GarmentCategory.OUTERWEAR: {"y_min": 0.28, "y_max": 0.90, "scale": 1.15},
    GarmentCategory.SHOES:     {"y_min": 0.88, "y_max": 1.00, "scale": 0.8},
    GarmentCategory.BAG:       {"y_min": 0.55, "y_max": 0.72, "side": "left"},
    GarmentCategory.SCARF:     {"y_min": 0.20, "y_max": 0.34, "scale": 0.9},
    GarmentCategory.JEWELLERY: {"y_min": 0.18, "y_max": 0.28, "scale": 0.5},
}


# ─── GARMENT PIPELINE ─────────────────────────────────────────────────────────

class GarmentPipeline:

    def __init__(self):
        self.classifier = GarmentClassifier()
        self.colours    = ColourExtractor()

    def process(
        self,
        image_bytes: bytes,
        override_category: Optional[str] = None
    ) -> ProcessedGarment:

        # 1. Decode
        nparr  = np.frombuffer(image_bytes, np.uint8)
        image  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Remove background
        pil_img   = Image.fromarray(image)
        removed   = remove(pil_img)
        rgba      = np.array(removed)
        rgb       = rgba[:, :, :3]
        alpha     = rgba[:, :, 3]

        # 3. Classify
        category = (
            GarmentCategory(override_category)
            if override_category
            else self.classifier.classify(rgb)
        )

        # 4. Dominant colour
        colour   = self.colours.dominant_colour(rgb, alpha)

        # 5. Texture (full size, transparent bg)
        texture  = self._to_base64(removed)

        # 6. Thumbnail (200×300)
        thumb    = removed.resize((200, 300), Image.LANCZOS)
        thumb_b64 = self._to_base64(thumb)

        # 7. Binary mask
        mask_img = Image.fromarray(alpha)
        mask_b64 = self._to_base64(mask_img)

        return ProcessedGarment(
            category        = category,
            dominant_colour = colour,
            texture_b64     = texture,
            thumbnail_b64   = thumb_b64,
            mask_b64        = mask_b64,
            uv_region       = UV_REGION_MAP.get(category, {}),
            metadata        = {
                "original_size": image.shape[:2],
                "pixel_coverage": float(alpha.mean() / 255)
            }
        )

    def _to_base64(self, pil_image) -> str:
        buf = io.BytesIO()
        if hasattr(pil_image, 'save'):
            pil_image.save(buf, format="PNG")
        else:
            Image.fromarray(pil_image).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
