# ─── SILHOUETTE AVATAR ENGINE ─────────────────────────────────────────────────
# Converts a single user photo into a parametric 3D body mesh

import cv2
import numpy as np
import mediapipe as mp
import trimesh
from PIL import Image
from rembg import remove
import base64
import io
import json
from dataclasses import dataclass
from typing import Tuple, Optional, List

mp_pose     = mp.solutions.pose
mp_face     = mp.solutions.face_detection
mp_segment  = mp.solutions.selfie_segmentation


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class BodyMeasurements:
    height_px:        float
    shoulder_width:   float
    chest_width:      float
    waist_width:      float
    hip_width:        float
    inseam_length:    float
    arm_length:       float
    neck_width:       float
    skin_tone:        Tuple[int, int, int]   # RGB
    # Normalised ratios (0.0–1.0) for mesh shaping
    shoulder_ratio:   float
    waist_ratio:      float
    hip_ratio:        float
    chest_ratio:      float


@dataclass
class AvatarMesh:
    vertices:   np.ndarray      # (N, 3) float32
    faces:      np.ndarray      # (F, 3) int32
    uvs:        np.ndarray      # (N, 2) float32
    normals:    np.ndarray      # (N, 3) float32
    skin_tone:  Tuple[int, int, int]
    measurements: BodyMeasurements


# ─── STEP 1: IMAGE PREPROCESSING ─────────────────────────────────────────────

class ImageProcessor:

    def __init__(self):
        self.pose      = mp_pose.Pose(
                            static_image_mode=True,
                            model_complexity=2,
                            enable_segmentation=True)
        self.face      = mp_face.FaceDetection(min_detection_confidence=0.5)
        self.segmenter = mp_segment.SelfieSegmentation(model_selection=1)

    def load_and_preprocess(self, image_bytes: bytes) -> np.ndarray:
        nparr  = np.frombuffer(image_bytes, np.uint8)
        image  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalise to 1024px tall for consistent landmark scaling
        h, w   = image.shape[:2]
        scale  = 1024 / h
        image  = cv2.resize(image, (int(w * scale), 1024))
        return image

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        pil_img    = Image.fromarray(image)
        removed    = remove(pil_img)      # rembg
        return np.array(removed)

    def extract_pose_landmarks(
        self, image: np.ndarray
    ) -> Optional[mp_pose.PoseLandmark]:
        results = self.pose.process(image)
        if not results.pose_landmarks:
            raise ValueError("No human body detected in image.")
        return results.pose_landmarks

    def extract_skin_tone(
        self, image: np.ndarray, landmarks
    ) -> Tuple[int, int, int]:
        h, w  = image.shape[:2]
        # Sample from face/neck region for accurate skin tone
        nose  = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nx, ny = int(nose.x * w), int(nose.y * h)
        # 20x20 sample around nose
        region = image[
            max(0, ny-10):min(h, ny+10),
            max(0, nx-10):min(w, nx+10)
        ]
        if region.size == 0:
            return (210, 180, 140)   # fallback neutral
        mean = region.mean(axis=(0, 1)).astype(int)
        return (int(mean[0]), int(mean[1]), int(mean[2]))


# ─── STEP 2: BODY MEASUREMENT ESTIMATION ─────────────────────────────────────

class MeasurementEstimator:

    # Anthropometric reference ratios (female, averaged)
    SHOULDER_TO_HEIGHT  = 0.259
    WAIST_TO_HEIGHT     = 0.181
    HIP_TO_HEIGHT       = 0.191
    CHEST_TO_HEIGHT     = 0.200
    INSEAM_TO_HEIGHT    = 0.471
    NECK_TO_SHOULDER    = 0.210

    def estimate(
        self,
        landmarks,
        image_shape: Tuple[int, int]
    ) -> BodyMeasurements:
        h, w = image_shape
        lm   = landmarks.landmark
        L    = mp_pose.PoseLandmark

        def px(landmark_id):
            pt = lm[landmark_id]
            return np.array([pt.x * w, pt.y * h])

        # Key points
        l_shoulder = px(L.LEFT_SHOULDER)
        r_shoulder = px(L.RIGHT_SHOULDER)
        l_hip      = px(L.LEFT_HIP)
        r_hip      = px(L.RIGHT_HIP)
        l_ankle    = px(L.LEFT_ANKLE)
        r_ankle    = px(L.RIGHT_ANKLE)
        l_wrist    = px(L.LEFT_WRIST)
        l_elbow    = px(L.LEFT_ELBOW)
        nose       = px(L.NOSE)

        # Raw pixel measurements
        shoulder_w  = np.linalg.norm(l_shoulder - r_shoulder)
        hip_w       = np.linalg.norm(l_hip - r_hip)
        body_top    = nose[1]
        body_bottom = (l_ankle[1] + r_ankle[1]) / 2
        height_px   = body_bottom - body_top
        inseam_px   = body_bottom - (l_hip[1] + r_hip[1]) / 2
        arm_px      = (
            np.linalg.norm(l_shoulder - l_elbow) +
            np.linalg.norm(l_elbow - l_wrist)
        )
        mid_body    = ((l_shoulder + r_shoulder) / 2 + (l_hip + r_hip) / 2) / 2
        # Waist estimated at midpoint between shoulder and hip
        waist_w     = shoulder_w * 0.72   # typical female ratio
        chest_w     = shoulder_w * 0.88
        neck_w      = shoulder_w * self.NECK_TO_SHOULDER

        # Normalised ratios for mesh deformation (around female average)
        # Values > 1.0 = wider than average, < 1.0 = narrower
        avg_shoulder = height_px * self.SHOULDER_TO_HEIGHT
        avg_hip      = height_px * self.HIP_TO_HEIGHT

        return BodyMeasurements(
            height_px      = height_px,
            shoulder_width = shoulder_w,
            chest_width    = chest_w,
            waist_width    = waist_w,
            hip_width      = hip_w,
            inseam_length  = inseam_px,
            arm_length     = arm_px,
            neck_width     = neck_w,
            skin_tone      = (0, 0, 0),   # filled by processor
            shoulder_ratio = float(shoulder_w / avg_shoulder),
            waist_ratio    = float(waist_w / (height_px * self.WAIST_TO_HEIGHT)),
            hip_ratio      = float(hip_w / avg_hip),
            chest_ratio    = float(chest_w / (height_px * self.CHEST_TO_HEIGHT)),
        )


# ─── STEP 3: PARAMETRIC BODY MESH GENERATION ─────────────────────────────────

class BodyMeshGenerator:
    """
    Builds a female parametric mesh from body measurements.
    Uses stacked elliptical cross-sections (like a proper
    parametric body model, but without SMPL licensing constraints).
    Each body segment is a tapered elliptic cylinder.
    """

    SEGMENTS = 32   # smoothness of cross-sections

    def generate(self, m: BodyMeasurements) -> AvatarMesh:
        vertices_list = []
        faces_list    = []
        uvs_list      = []

        # Normalise everything to unit height (2.0 Three.js units)
        scale = 2.0 / m.height_px

        def sw(px): return px * scale   # scale width
        def sh(px): return px * scale   # scale height

        shoulder_r = sw(m.shoulder_width) / 2
        chest_r    = sw(m.chest_width)    / 2
        waist_r    = sw(m.waist_width)    / 2
        hip_r      = sw(m.hip_width)      / 2
        neck_r     = sw(m.neck_width)     / 2
        head_r     = neck_r * 1.85

        # Y positions (bottom = 0, top = 2.0)
        y_feet     = 0.0
        y_knee     = sh(m.inseam_length * 0.48)
        y_hip      = sh(m.inseam_length)
        y_waist    = y_hip  + sh(m.height_px * 0.08)
        y_chest    = y_waist + sh(m.height_px * 0.13)
        y_shoulder = y_chest + sh(m.height_px * 0.07)
        y_neck_bot = y_shoulder + sh(m.height_px * 0.03)
        y_neck_top = y_neck_bot + sh(m.height_px * 0.05)
        y_head_bot = y_neck_top
        y_head_top = 2.0

        # Body segments: list of (y_bot, r_bot_x, r_bot_z, y_top, r_top_x, r_top_z)
        # x-radius = width, z-radius = depth (depth ≈ 0.6× width for female form)
        DZ = 0.62   # depth ratio

        torso_segments = [
            # (y_bot, rx_bot, rz_bot, y_top, rx_top, rz_top, label)
            (y_feet,     hip_r*0.28,      hip_r*0.28*DZ,
             y_knee,     hip_r*0.30,      hip_r*0.30*DZ,     "l_calf"),
            (y_feet,     hip_r*0.28,      hip_r*0.28*DZ,
             y_knee,     hip_r*0.30,      hip_r*0.30*DZ,     "r_calf"),
            (y_knee,     hip_r*0.30,      hip_r*0.30*DZ,
             y_hip,      hip_r*0.42,      hip_r*0.42*DZ,     "l_thigh"),
            (y_knee,     hip_r*0.30,      hip_r*0.30*DZ,
             y_hip,      hip_r*0.42,      hip_r*0.42*DZ,     "r_thigh"),
            (y_hip,      hip_r,           hip_r*DZ,
             y_waist,    waist_r,         waist_r*DZ,         "lower_torso"),
            (y_waist,    waist_r,         waist_r*DZ,
             y_chest,    chest_r,         chest_r*DZ,         "mid_torso"),
            (y_chest,    chest_r,         chest_r*DZ,
             y_shoulder, shoulder_r,      shoulder_r*DZ,      "upper_torso"),
            (y_neck_bot, neck_r,          neck_r,
             y_neck_top, neck_r*0.92,     neck_r*0.92,        "neck"),
        ]

        vertex_offset = 0

        def add_elliptic_cylinder(
            y_bot, rx_b, rz_b,
            y_top, rx_t, rz_t,
            x_offset=0.0
        ):
            nonlocal vertex_offset
            n     = self.SEGMENTS
            verts = []
            uvs   = []

            for i in range(n):
                angle     = 2 * np.pi * i / n
                cos_a     = np.cos(angle)
                sin_a     = np.sin(angle)
                # Bottom ring
                verts.append([x_offset + rx_b*cos_a, y_bot, rz_b*sin_a])
                uvs.append(  [i/n,                    0.0])
                # Top ring
                verts.append([x_offset + rx_t*cos_a, y_top, rz_t*sin_a])
                uvs.append(  [i/n,                    1.0])

            faces = []
            for i in range(n):
                b0 = vertex_offset + i*2
                b1 = vertex_offset + ((i+1) % n)*2
                t0 = b0 + 1
                t1 = b1 + 1
                faces.append([b0, t0, b1])
                faces.append([b1, t0, t1])

            vertices_list.append(np.array(verts,  dtype=np.float32))
            faces_list.append(   np.array(faces,  dtype=np.int32))
            uvs_list.append(     np.array(uvs,    dtype=np.float32))
            vertex_offset += len(verts)

        # Torso (centred)
        for seg in torso_segments[4:]:
            add_elliptic_cylinder(seg[0],seg[1],seg[2],seg[3],seg[4],seg[5])

        # Legs (offset left/right)
        leg_offset = hip_r * 0.38
        for seg in torso_segments[:2]:
            add_elliptic_cylinder(
                seg[0],seg[1],seg[2],seg[3],seg[4],seg[5],
                x_offset=-leg_offset
            )
            add_elliptic_cylinder(
                seg[0],seg[1],seg[2],seg[3],seg[4],seg[5],
                x_offset=leg_offset
            )
        for seg in torso_segments[2:4]:
            add_elliptic_cylinder(
                seg[0],seg[1],seg[2],seg[3],seg[4],seg[5],
                x_offset=-leg_offset*0.7
            )
            add_elliptic_cylinder(
                seg[0],seg[1],seg[2],seg[3],seg[4],seg[5],
                x_offset=leg_offset*0.7
            )

        # Arms
        arm_r_top = shoulder_r * 0.22
        arm_r_bot = shoulder_r * 0.14
        arm_len   = sh(m.arm_length)
        arm_y_top = y_shoulder
        arm_y_bot = y_shoulder - arm_len

        add_elliptic_cylinder(
            arm_y_bot, arm_r_bot, arm_r_bot*0.85,
            arm_y_top, arm_r_top, arm_r_top*0.85,
            x_offset=-(shoulder_r + arm_r_top*0.5)
        )
        add_elliptic_cylinder(
            arm_y_bot, arm_r_bot, arm_r_bot*0.85,
            arm_y_top, arm_r_top, arm_r_top*0.85,
            x_offset= (shoulder_r + arm_r_top*0.5)
        )

        # Head — sphere approximated via stacked elliptic rings
        head_h    = y_head_top - y_head_bot
        head_segs = 14
        for i in range(head_segs):
            t0  = i / head_segs
            t1  = (i+1) / head_segs
            ang0 = np.pi * t0
            ang1 = np.pi * t1
            r0  = head_r * np.sin(ang0) * 1.0
            r1  = head_r * np.sin(ang1) * 1.0
            rx0 = r0 * 0.88   # slightly narrower face
            rx1 = r1 * 0.88
            add_elliptic_cylinder(
                y_head_bot + t0*head_h, rx0, r0,
                y_head_bot + t1*head_h, rx1, r1,
            )

        # Compile
        all_verts   = np.vstack(vertices_list)
        all_faces   = np.vstack(faces_list)
        all_uvs     = np.vstack(uvs_list)

        # Compute normals
        mesh        = trimesh.Trimesh(
                        vertices=all_verts,
                        faces=all_faces,
                        process=False
                      )
        mesh.fix_normals()
        normals     = mesh.vertex_normals.astype(np.float32)

        return AvatarMesh(
            vertices     = all_verts,
            faces        = all_faces,
            uvs          = all_uvs,
            normals      = normals,
            skin_tone    = m.skin_tone,
            measurements = m
        )

    def to_gltf_dict(self, avatar: AvatarMesh) -> dict:
        """Export as GLTF-compatible JSON for Three.js consumption."""
        r, g, b = [c/255.0 for c in avatar.skin_tone]
        return {
            "vertices":   avatar.vertices.tolist(),
            "faces":      avatar.faces.tolist(),
            "uvs":        avatar.uvs.tolist(),
            "normals":    avatar.normals.tolist(),
            "skin_tone":  {"r": r, "g": g, "b": b},
            "measurements": {
                "shoulder_ratio": avatar.measurements.shoulder_ratio,
                "waist_ratio":    avatar.measurements.waist_ratio,
                "hip_ratio":      avatar.measurements.hip_ratio,
                "chest_ratio":    avatar.measurements.chest_ratio,
            }
        }


# ─── MASTER AVATAR PIPELINE ───────────────────────────────────────────────────

class AvatarPipeline:

    def __init__(self):
        self.processor   = ImageProcessor()
        self.estimator   = MeasurementEstimator()
        self.generator   = BodyMeshGenerator()

    def run(self, image_bytes: bytes) -> dict:
        # 1. Load
        image       = self.processor.load_and_preprocess(image_bytes)
        # 2. Landmarks
        landmarks   = self.processor.extract_pose_landmarks(image)
        # 3. Skin tone
        skin_tone   = self.processor.extract_skin_tone(image, landmarks)
        # 4. Measurements
        measurements         = self.estimator.estimate(
                                landmarks, image.shape[:2])
        measurements.skin_tone = skin_tone
        # 5. Mesh
        avatar      = self.generator.generate(measurements)
        # 6. Export
        return self.generator.to_gltf_dict(avatar)
