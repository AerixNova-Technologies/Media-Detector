from __future__ import annotations
import os
import cv2
import numpy as np
import json
import logging

log = logging.getLogger("face_rec")

try:
    from deepface import DeepFace as _DF
    from scipy.spatial.distance import cosine
    _DEEPFACE_OK = True
except Exception:
    _DEEPFACE_OK = False

class FaceRecognizer:
    def __init__(self, db_path: str = None, skip_frames: int = 15, backend: str = "opencv", model_name="Facenet512"):
        self.skip_frames = skip_frames
        self.backend = backend
        self.model_name = model_name
        self.db_path = db_path
        self._cache: dict[int, dict] = {} # Changed to store dict results
        self._counters: dict[int, int] = {}
        self._available = _DEEPFACE_OK
        self._known_faces: list[dict] = [] # List of {"id": int, "name": str, "encoding": list[float]}
        
        # Stricter threshold to prevent false positives (0.4 is the max, 0.25-0.3 is safer)
        self.threshold = 0.35 
        # Minimum size of face crop to attempt recognition
        self.min_face_size = 20 

        if db_path and self._available:
            self.load_from_folder(db_path)

    def load_from_folder(self, db_path: str):
        """Scan subfolders of db_path and load face signatures."""
        if not db_path:
            return
        self.db_path = db_path
            
        from app.db.session import get_db_connection
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            log.info("FaceRec: Scanning %s for staff faces...", db_path)
            self._known_faces = [] # Clear existing list to avoid duplicates on reload
            for person_name in os.listdir(db_path):
                if person_name.lower() in ["branding", "snapshots"]:
                    continue
                person_dir = os.path.join(db_path, person_name)
                if not os.path.isdir(person_dir):
                    continue
                
                # Fetch staff_id and db_id for this person
                cur.execute("SELECT id, staff_id FROM staff_profiles WHERE name = %s", (person_name,))
                row = cur.fetchone()
                db_id = row['id'] if row else None
                display_id = row['staff_id'] if row else ""

                # Load all valid images from the directory
                loaded_count = 0
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_dir, filename)
                        try:
                            emb = self.extract_embedding(cv2.imread(img_path))
                            if emb:
                                self._known_faces.append({
                                    "id": db_id,
                                    "name": person_name, 
                                    "display_id": display_id, 
                                    "encoding": emb
                                })
                                loaded_count += 1
                        except Exception as e:
                            log.warning("FaceRec: Failed to load %s: %s", img_path, e)
                
                if loaded_count > 0:
                    log.info("FaceRec: Loaded staff member: %s (ID: %s, %d photos)", person_name, display_id, loaded_count)
        except Exception as e:
            log.error("FaceRec: Error loading from folder: %s", e)
        finally:
            if conn: conn.close()

    def set_known_faces(self, faces: list[dict]):
        """Load biometric data directly from memory/DB."""
        self._known_faces = faces

    def extract_embedding(self, image_input) -> list[float] | None:
        """Helper to get a face signature (embedding) from raw bytes or numpy array."""
        if not self._available: return None
        try:
            # If bytes, convert to numpy
            if isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = image_input

            if img is None or img.size == 0:
                return None

            # Check if the face is large enough for a reliable measurement
            h, w = img.shape[:2]
            if h < self.min_face_size or w < self.min_face_size:
                return None

            result = _DF.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend=self.backend,
                enforce_detection=False,
                align=True
            )
            if result and len(result) > 0:
                return result[0]["embedding"]
        except Exception:
            pass
        return None

    def recognize(self, face_crop: np.ndarray, track_id: int) -> dict:
        """Compare current face embedding against known list in memory."""
        default_res = {"id": None, "name": "Unknown", "display_id": ""}
        if not self._available or face_crop is None or face_crop.size == 0 or not self._known_faces:
            return default_res

        counter = self._counters.get(track_id, 0)
        self._counters[track_id] = counter + 1

        # We only run recognition every N frames to save CPU, but we return the cached dict in between
        if counter % self.skip_frames != 0 and track_id in self._cache:
            return self._cache[track_id]

        # Check image quality/size before proceeding
        h, w = face_crop.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            self._cache[track_id] = default_res
            return default_res

        # Extract current embedding
        current_enc = self.extract_embedding(face_crop)
        if not current_enc:
            self._cache[track_id] = default_res
            return default_res

        # Find best match in memory
        best_id = None
        best_name = "Unknown"
        best_display_id = ""
        best_score = self.threshold

        for identity in self._known_faces:
            dist = cosine(current_enc, identity["encoding"])
            if dist < best_score:
                best_score = dist
                best_id = identity.get("id")
                best_name = identity["name"]
                best_display_id = identity.get("display_id", "")

        res = {"id": best_id, "name": best_name, "display_id": best_display_id}
        self._cache[track_id] = res
        return res

    def purge(self, active_ids: set[int]) -> None:
        """Remove cached entries for disappeared tracks."""
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                self._cache.pop(tid, None)
                self._counters.pop(tid, None)
