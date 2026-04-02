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
        self.backend = backend # Use passed backend (default opencv)
        self.model_name = "Facenet512"
        self.db_path = db_path
        self._cache: dict[int, dict] = {} # Changed to store dict results
        self._counters: dict[int, int] = {}
        self._available = _DEEPFACE_OK
        self._known_faces: list[dict] = [] # List of {"id": int, "name": str, "encoding": list[float]}
        
        # ULTRA-STRICT threshold to avoid misidentification (was 0.50)
        self.threshold = 0.38 
        self.margin = 0.06
        # Minimum size of face crop to attempt recognition
        self.min_face_size = 10 

        if db_path and self._available:
            self.load_from_folder(db_path)

    def load_from_folder(self, db_path: str):
        """Scan subfolders and load averaged face signatures per person."""
        if not db_path: return
        self.db_path = db_path
            
        from app.db.session import get_db_connection
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            print(f"DEBUG: FaceRec: Scanning {db_path} for staff faces...")
            log.info("FaceRec: Scanning %s for staff faces...", db_path)
            loaded_faces = []
            
            for person_name in os.listdir(db_path):
                if person_name.lower() in ["branding", "snapshots", "movement"]:
                    continue
                person_dir = os.path.join(db_path, person_name)
                if not os.path.isdir(person_dir):
                    continue
                
                # Case-insensitive DB lookup (Including newly added Category)
                cur.execute("SELECT id, name, staff_id, category FROM staff_profiles WHERE LOWER(name) = LOWER(%s)", (person_name,))
                row = cur.fetchone()
                if not row:
                    log.warning("FaceRec: No database profile found for folder '%s'. Skipping.", person_name)
                    continue

                db_id = row['id']
                db_name = row['name']
                display_id = row['staff_id'] or ""
                category = row.get('category', 'Staff')

                person_embs = []
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_dir, filename)
                        try:
                            img = cv2.imread(img_path)
                            if img is None: continue
                            # For loading known faces, we don't need to enforce detection or skip
                            # as we are processing full images and want detection to happen.
                            emb = self.extract_embedding(img, enforce_detection=False)
                            if emb:
                                person_embs.append(emb)
                        except Exception:
                            continue
                
                if person_embs:
                    avg_emb = np.mean(person_embs, axis=0).tolist()
                    loaded_faces.append({
                        "id": db_id,
                        "name": person_name, 
                        "display_id": display_id, 
                        "category": category, # New
                        "encoding": avg_emb
                    })
                    log.info("FaceRec: Loaded staff member: %s [Category: %s] (%d photos)", db_name, category, len(person_embs))
            
            # ATOMIC SWAP: No transient empty states!
            self._known_faces = loaded_faces
            # self._cache.clear() # REMOVED: Keep current track identities stable during reload
            # self._counters.clear()
                    
        except Exception as e:
            log.error("FaceRec: Error loading from folder: %s", e)
        finally:
            if conn: conn.close()

    def set_known_faces(self, faces: list[dict]):
        """Load biometric data directly from memory/DB."""
        self._known_faces = faces

    def extract_embedding(self, image_input, enforce_detection: bool = True, detector_backend: str = None) -> list[float] | None:
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

            # For live recognition, we check size. For staff upload, we are more lenient.
            h, w = img.shape[:2]
            if enforce_detection and (h < self.min_face_size or w < self.min_face_size):
                return None

            log.info(f"FaceRec: Starting represent (Backend={detector_backend or self.backend})...")
            result = _DF.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend=detector_backend or self.backend,
                enforce_detection=enforce_detection, 
                align=True
            )
            log.info(f"FaceRec: Represent finished (Result ok={bool(result)})")
            if result and len(result) > 0:
                return result[0]["embedding"]
        except Exception as e:
            log.warning(f"FaceRec: extract_embedding error: {e}")
        return None

    def recognize(self, face_crop: np.ndarray, track_id: int) -> dict:
        """Compare current face embedding against known list in memory."""
        default_res = {"id": None, "name": "Unknown", "display_id": ""}
        if not self._available or face_crop is None or face_crop.size == 0 or not self._known_faces:
            if not self._known_faces:
                log.warning(f"FaceRec: Cannot recognize Track {track_id} - Known database is empty!")
            elif not self._available:
                log.warning("FaceRec: Engine not available (DeepFace load failed)")
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

        # Extract current embedding (Relaxed detection since we already cropped)
        current_enc = self.extract_embedding(face_crop, enforce_detection=False)
        if current_enc is None:
            log.warning(f"FaceRec: Track {track_id} - Could not extract embedding from face crop.")
            return {"id": None, "name": "Unknown", "display_id": "Unrecognized"}

        # Find best match in memory
        best_id = None
        best_name = "Unknown"
        best_display_id = ""
        best_dist = float('inf')
        second_best_dist = float('inf')
        
        # We'll use a margin (e.g. 0.05) to ensure the match is distinct
        # If the gap between top-1 and top-2 is too small, it's ambiguous.
        CONFIDENCE_MARGIN = 0.05

        for identity in self._known_faces:
            dist = cosine(current_enc, identity["encoding"])
            if dist < best_dist:
                second_best_dist = best_dist
                best_dist = dist
                best_id = identity.get("id")
                best_name = identity["name"]
                best_display_id = identity.get("display_id", "")
            elif dist < second_best_dist:
                second_best_dist = dist
        
        # DEBUG LOG: See how close we are to a match
        if best_name != "Unknown":
            log.info(f"FaceRec: Best match for Track {track_id} is '{best_name}' with dist {best_dist:.3f} (Threshold: {self.threshold:.2f})")

        # ULTRA-STRICT Security Gate:
        # 1. Must be below threshold (0.38 - Very high confidence)
        # 2. Must be DISTINCT (Significant gap between Top 1 and Top 2)
        # 3. Handle ambiguity specifically
        
        confidence_gap = second_best_dist - best_dist
        is_ambiguous = confidence_gap < 0.05 # REQUIRE at least 0.05 gap for any person match
        
        if is_ambiguous and best_dist < self.threshold:
             log.warning(f"FaceRec: Ambiguity detected between candidates (Gap: {confidence_gap:.3f}). Rejecting match for Track {track_id}")
             res = default_res # Force Unknown if unclear
        elif best_dist < self.threshold and (not is_ambiguous or best_dist < 0.30):
            # If extremely confident (dist < 0.30), we can be slightly less strict with the gap
            res = {"id": best_id, "name": best_name, "display_id": best_display_id}
            log.info(f"FaceRec: Recognized {best_name} (dist={best_dist:.3f}, gap={confidence_gap:.3f})")
        else:
            log.info(f"FaceRec: No match for Track {track_id} (best: {best_name} at {best_dist:.3f}, gap={confidence_gap:.3f})")
            res = default_res
            
        self._cache[track_id] = res
        return res

    def purge(self, active_ids: set[int]) -> None:
        """Remove cached entries for disappeared tracks."""
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                self._cache.pop(tid, None)
                self._counters.pop(tid, None)
