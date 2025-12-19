import os
import glob
import cv2
import torch
import numpy as np
import time
import threading
import queue
import uuid
import base64
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# --- EXTERNAL DEPENDENCIES ---
# Optimize PyTorch CPU Threading
torch.set_num_threads(4) # Limit to 4 threads to prevent fighting with Video loop

# Ensure database.py is present in the directory
try:
    import database as db
except ImportError:
    print("⚠️ WARNING: database.py not found. Database features will fail.")

# ==========================================
# CLASS: FACE ANALYZER
# Handles Neural Networks (YOLO + FaceNet)
# ==========================================
class FaceAnalyzer:
    def __init__(self, model_path='yolov8n-face.mlpackage'):
        self.MODEL_PATH = model_path
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"⚡ Loading FaceNet (PyTorch) on: {self.device}")

        # Load FaceNet512
        self.recognizer = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device)
        self.recognizer.eval()

        print("🧠 Loading YOLO Models...")
        # 1. TRACKER: For high-speed video loop (CPU/CoreML)
        # Note: task='pose' is explicitly set to ensure CoreML metadata is respected
        self.tracker = YOLO(self.MODEL_PATH, task='pose') if os.path.exists(self.MODEL_PATH) else YOLO('yolov8n.pt')
        if os.path.exists(self.MODEL_PATH):
            print(f"✅ Success: Loaded Face Model ({self.MODEL_PATH})")
        else:
            print(f"⚠️ WARNING: '{self.MODEL_PATH}' not found! Loaded Standard YOLO (Objects) instead.")
        # 2. VALIDATOR: For quality checks & reloading (Separate instance to avoid locking)
        self.validator = YOLO(self.MODEL_PATH, task='pose') if os.path.exists(self.MODEL_PATH) else YOLO('yolov8n.pt')

    def get_embedding(self, face_img, keypoints=None):
        """
        Aligns face (if keypoints provided), adds black bars, then embeds.
        """
        try:
            # --- GEOMETRIC ALIGNMENT (NEW) ---
            if keypoints is not None:
                # Expecting keypoints in original image coordinates corresponding to face_img
                # Keypoints: 0: Left Eye, 1: Right Eye, 2: Nose, 3: Left Mouth, 4: Right Mouth
                if len(keypoints) >= 2:
                    left_eye = keypoints[0]
                    right_eye = keypoints[1]
                    
                    # Calculate Angle
                    dy = right_eye[1] - left_eye[1]
                    dx = right_eye[0] - left_eye[0]
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    # Rotate Image
                    # Get center of face
                    h, w = face_img.shape[:2]
                    center = (w // 2, h // 2)
                    
                    # Compute Rotation Matrix
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # Perform Rotation
                    face_img = cv2.warpAffine(face_img, M, (w, h), flags=cv2.INTER_CUBIC)
            # -----------------------------------

            h, w = face_img.shape[:2]

            # --- SQUARE CROP LOGIC ---
            if h != w:
                diff = abs(h - w)
                top, bottom, left, right = 0, 0, 0, 0

                if h < w:  # Image is too wide, add height
                    top = diff // 2
                    bottom = diff - top
                else:  # Image is too tall, add width
                    left = diff // 2
                    right = diff - left

                face_img = cv2.copyMakeBorder(face_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # -----------------------------------

            face_img = cv2.resize(face_img, (160, 160))
            # FIXED NORMALIZATION: FaceNet (vggface2) expects [-1, 1] approx, or whitening.
            # (x - 127.5) / 128.0 is the standard fixed normalization for FaceNet.
            face_img = (np.float32(face_img) - 127.5) / 128.0
            face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.recognizer(face_tensor)
                # L2 NORMALIZE (CRITICAL for Euclidean Distance)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding[0]
        except Exception:
            return None

    def check_face_quality(self, img_bgr):
        """
        Validates the image for:
        1. Blur (Laplacian var)
        2. Face Count (Must be 1)
        3. Pose (Yaw/Roll using Keypoints)
        Returns: (is_valid, reason, face_crop)
        """
        if img_bgr is None:
            return False, "No image provided", None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. BLUR CHECK
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"🔍 DEBUG: Blur Var: {blur_var:.2f}")
        if blur_var < 50:  # Threshold for "Clear" (Relaxed further)
            return False, f"Image is too blurry ({blur_var:.1f}). Please hold steady.", None

        # 2. RUN DETECTION (Use VALIDATOR)
        # We are in the API thread here, so we should use VALIDATOR to avoid race with TRACKER in detection thread
        results = self.validator.predict(img_bgr, verbose=False, conf=0.5, device='cpu')
        
        if not results or not results[0].boxes:
            print("🔍 DEBUG: No face detected.")
            return False, "No face detected.", None
            
        boxes = results[0].boxes
        if len(boxes) > 1:
            print(f"🔍 DEBUG: Faces found: {len(boxes)}")
            return False, "Multiple faces detected. Please be alone.", None
            
        # 3. EXTRACT FACE & KEYPOINTS
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = img_bgr[max(0, y1):min(img_bgr.shape[0], y2),
                            max(0, x1):min(img_bgr.shape[1], x2)]
                            
        # 4. POSE CHECK (Using 5-point Keypoints)
        # 0: Left Eye, 1: Right Eye, 2: Nose, 3: Left Mouth, 4: Right Mouth
        if results[0].keypoints is not None and results[0].keypoints.xy is not None:
             # Check confidence
             if results[0].keypoints.conf is not None:
                 conf = results[0].keypoints.conf[0].cpu().numpy()
                 print(f"🔍 DEBUG: Keypoint Conf: {conf.mean():.2f}")
                 if conf.mean() < 0.3:
                      return False, "Face not clear enough (low keypoint confidence).", None

             kpts = results[0].keypoints.xy[0].cpu().numpy()
             if len(kpts) >= 3:
                 # A. ROLL (Tilt) - Angle between eyes
                 le = kpts[0]
                 re = kpts[1]
                 dy = re[1] - le[1]
                 dx = re[0] - le[0]
                 angle = np.degrees(np.arctan2(dy, dx))
                 print(f"🔍 DEBUG: Head Tilt: {angle:.2f}")
                 if abs(angle) > 25: # Relaxed from 15
                     return False, "Head tilted. Please keep head straight.", None
                     
                 # B. YAW (Turn) - Distance from Nose to Eyes
                 nose = kpts[2]
                 dist_l_nose = np.linalg.norm(nose - le) # Nose to Left Eye
                 dist_r_nose = np.linalg.norm(nose - re) # Nose to Right Eye
                 
                 # Avoid division by zero
                 if dist_l_nose > 0 and dist_r_nose > 0:
                     ratio = max(dist_l_nose, dist_r_nose) / min(dist_l_nose, dist_r_nose)
                     print(f"🔍 DEBUG: Turn Ratio: {ratio:.2f}")
                     # Frontal face should have ratio close to 1.0. 
                     # Side face will have large ratio.
                     if ratio > 4.0: # Relaxed from 2.0
                         return False, "Face turned. Please look straight at camera.", None

        return True, "Quality OK", face_crop


# ==========================================
# CLASS: FACE DATABASE
# Handles loading gallery, caching, and matching
# ==========================================
class FaceDatabase:
    def __init__(self, db_path="gallery", confidence_threshold=0.55, device=None):
        self.DB_PATH = db_path
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.face_db_embeddings = []
        self.face_db_names = []
        
        # Cache file path
        self.CACHE_PATH = os.path.join(self.DB_PATH, "vectors.pt")

    def reload_database(self, face_analyzer, class_id_or_code=None):
        """
        Loads gallery images for a SPECIFIC CLASS.
        Structure: gallery/{class_id}/vectors/{name}.pt
        """
        if not class_id_or_code:
            print("⚠️ No Class ID provided. Clearing memory.")
            self.face_db_embeddings = []
            self.face_db_names = []
            return

        target_dir = os.path.join(self.DB_PATH, str(class_id_or_code))
        vectors_dir = os.path.join(target_dir, "vectors")
        legacy_cache_path = os.path.join(target_dir, "vectors.pt")
        
        print(f"📂 Indexing Gallery for Class: {class_id_or_code}...")
        
        # 1. Ensure Directory Exists
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # -----------------------------------------------
        # MIGRATION: Split Legacy vectors.pt -> vectors/*.pt
        # -----------------------------------------------
        if os.path.exists(legacy_cache_path):
            print("📦 Found legacy monolithic cache. Migrating to granular files...")
            try:
                if not os.path.exists(vectors_dir):
                     os.makedirs(vectors_dir)
                     
                legacy_data = torch.load(legacy_cache_path)
                for name, embedding in legacy_data.items():
                    safe_path = os.path.join(vectors_dir, f"{name}.pt")
                    torch.save(embedding, safe_path)
                    print(f"   -> Migrated {name}")
                
                # Backup and remove legacy
                os.rename(legacy_cache_path, legacy_cache_path + ".bak")
                print("✅ Migration complete. Legacy cache backed up.")
            except Exception as e:
                print(f"⚠️ Migration failed: {e}")
        # -----------------------------------------------

        if not os.path.exists(vectors_dir):
            os.makedirs(vectors_dir)

        # 2. Gather All Names from Images (Source of Truth)
        gallery_images = glob.glob(os.path.join(target_dir, "*.jpg"))
        gallery_images += glob.glob(os.path.join(target_dir, "*.png"))
        
        new_names = []
        new_embeddings = []
        
        processed_count = 0
        loaded_count = 0

        for img_path in gallery_images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            vector_path = os.path.join(vectors_dir, f"{name}.pt")
            
            embedding = None
            
            # A. Try Load from Granular Cache
            if os.path.exists(vector_path):
                try:
                    embedding = torch.load(vector_path)
                    loaded_count += 1
                except:
                    print(f"⚠️ Corrupt vector file for {name}, regenerating...")

            # B. If No Cache, Regenerate
            if embedding is None:
                try:
                    processed_count += 1
                    img = cv2.imread(img_path)
                    if img is None: continue
                    
                    # Optimization: Resize huge images
                    h, w = img.shape[:2]
                    if w > 640:
                         scale = 640 / w
                         img = cv2.resize(img, (640, int(h * scale)))

                    # 1. Detect
                    results = face_analyzer.validator.predict(img, verbose=False, conf=0.15, device='cpu')
                    
                    face_crop = None
                    if results and results[0].boxes:
                        largest_area = 0
                        best_box = None
                        best_idx = -1
                        for i, box in enumerate(results[0].boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)
                            if area > largest_area:
                                largest_area = area
                                best_box = (x1, y1, x2, y2)
                                best_idx = i # TRACK INDEX
                        
                        if best_box:
                            x1, y1, x2, y2 = best_box
                            face_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

                    if face_crop is None or face_crop.size == 0:
                        face_crop = img # Fallback

                    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    
                    # ALIGNMENT: Pass Keypoints if available
                    keypoints_for_align = None
                    if results and results[0].keypoints is not None:
                         # Keypoints are on the FULL image 'img'
                         # But 'face_crop' is a slice.
                         # FaceAnalyzer.get_embedding expects crop.
                         # AND alignment logic expects keypoints relative to that crop.
                         
                         if best_box and best_idx != -1:
                             bx1, by1, bx2, by2 = best_box
                             # Use best_idx to get the CORRECT keypoints
                             raw_kpts = results[0].keypoints.xy[best_idx].cpu().numpy() 
                             
                             # Adjust keypoints to be relative to the CROP
                             keypoints_for_align = []
                             for (kx, ky) in raw_kpts:
                                 keypoints_for_align.append((kx - bx1, ky - by1))
                                 
                    is_aligned = "Aligned" if keypoints_for_align is not None else "Unaligned"
                    embedding = face_analyzer.get_embedding(face_crop, keypoints=keypoints_for_align)
                    
                    if embedding is not None:
                        # Moves to CPU for storage
                        embedding = embedding.cpu()
                        # Save Granular Cache
                        torch.save(embedding, vector_path)
                        print(f"   -> Generated & Saved {name}.pt ({is_aligned})")
                        
                except Exception as e:
                    print(f"⚠️ Error processing {name}: {e}")

            if embedding is not None:
                # --- SYNC ENROLLMENT (Fix for Manual Moves) ---
                if class_id_or_code and str(class_id_or_code).isdigit():
                     try:
                         # 1. Ensure Student Exists (using Name as ID)
                         db.add_student(name, name, img_path)
                         # 2. Enroll
                         db.enroll_student(name, subject_id=int(class_id_or_code))
                     except Exception as e:
                         pass
                # ---------------------------------------------

                new_embeddings.append(embedding)
                new_names.append(name)
        
        # 3. Build Final Tensor (In Memory)
        if new_embeddings:
            self.face_db_embeddings = torch.stack(new_embeddings).to(self.device)
            self.face_db_names = new_names
            print(f"✅ Class {class_id_or_code} Loaded. Active Faces: {len(self.face_db_names)} (Cached: {loaded_count}, New: {processed_count})")
        else:
            self.face_db_embeddings = []
            self.face_db_names = []
            print(f"⚠️ Class {class_id_or_code} is empty.")

    def get_match(self, target_embedding):
        """
        Returns (name, score_str) for the best match.
        """
        name = "Unknown"
        score_str = ""

        if target_embedding is not None and len(self.face_db_embeddings) > 0:
            # Vectorized Euclidean Distance
            # (embeddings - target)^2
            diff = self.face_db_embeddings - target_embedding
            dist = torch.norm(diff, dim=1).cpu().numpy() # Transfer only results to CPU
            best_idx = np.argmin(dist)
            min_dist = dist[best_idx]

            if min_dist < self.CONFIDENCE_THRESHOLD: 
                name = self.face_db_names[best_idx]
                score_str = f"({min_dist:.2f})"
            print(f"🕵️ OPTIONAL LOG: Closest: {self.face_db_names[best_idx]} ({min_dist:.4f})")
        
        return name, score_str
        
    def find_nearest_dist(self, target_embedding):
        """Returns the distance to the nearest face in DB."""
        if target_embedding is None or len(self.face_db_embeddings) == 0:
            return float('inf'), None
            
        diff = self.face_db_embeddings - target_embedding
        dist = torch.norm(diff, dim=1).cpu().numpy()
        min_idx = np.argmin(dist)
        return dist[min_idx], self.face_db_names[min_idx]

    def add_face_to_memory(self, name, embedding, class_id):
        """
        Directly appends a new face to the running memory & updates GRANULAR cache.
        """
        if embedding is None: return

        print(f"⚡ Hot-adding {name} to memory (Class {class_id})...")
        
        # 1. Update In-Memory Tensor
        emb_tensor = embedding.to(self.device)
        if len(self.face_db_embeddings) == 0:
            self.face_db_embeddings = emb_tensor.unsqueeze(0)
        else:
            self.face_db_embeddings = torch.cat((self.face_db_embeddings, emb_tensor.unsqueeze(0)))
            
        self.face_db_names.append(name)
        
        # 2. Update Disk Cache (Granular)
        target_dir = os.path.join(self.DB_PATH, str(class_id))
        vectors_dir = os.path.join(target_dir, "vectors")
        
        if not os.path.exists(vectors_dir):
            os.makedirs(vectors_dir)
            
        vector_path = os.path.join(vectors_dir, f"{name}.pt")
        
        try:
            torch.save(embedding.cpu(), vector_path)
            print(f"💾 Saved vector: {name}.pt")
        except Exception as e:
             print(f"⚠️ Auto-save cache failed: {e}")


# ==========================================
# CLASS: REGISTRATION MANAGER
# Handles new user staging and saving
# ==========================================
class RegistrationManager:
    def __init__(self, db_path="gallery"):
        self.registration_cache = {}  # {token: {img, embedding, ...}}
        self.DB_PATH = db_path

    def validate_and_stage(self, img_bgr, face_analyzer, face_db):
        """
        Checks quality, doubles check dupes, generates temp key.
        Returns dict with status.
        """
        valid, reason, face_crop = face_analyzer.check_face_quality(img_bgr)
        
        if not valid:
            return {"status": "error", "reason": reason}
            
        # 5. DUPLICATE CHECK (Using Embedding)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        embedding = face_analyzer.get_embedding(face_rgb)
        
        warning = ""
        
        min_dist, potential_name = face_db.find_nearest_dist(embedding)
        if min_dist < 0.5:
            warning = f"Warning: You look very similar to '{potential_name}'."

        # 6. STAGE IN CACHE
        token = str(uuid.uuid4())
        
        # Prepare Preview (Base64)
        _, buffer = cv2.imencode('.jpg', face_crop)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        preview_data = f"data:image/jpeg;base64,{b64_img}"
        
        # Store embedding (move to CPU to save GPU mem)
        emb_cpu = embedding.cpu() if embedding is not None else None
        
        self.registration_cache[token] = {
            "full_img": img_bgr,
            "embedding": emb_cpu,
            "timestamp": time.time()
        }
        
        return {
            "status": "ok",
            "token": token, 
            "message": "Face quality is good.",
            "warning": warning,
            "preview": preview_data
        }

    def commit_registration(self, token, name, face_db, class_id):
        """
        Moves from cache to gallery/{class_id}/. Returns file path.
        """
        if token not in self.registration_cache:
            return {"success": False, "message": "Session expired or invalid token."}
            
        data = self.registration_cache.pop(token)
        img = data['full_img']
        embedding = data['embedding']
        
        # Save to Gallery/Class_ID
        safe_name = "".join(x for x in name if x.isalnum() or x in " _-").strip()
        class_based_path = os.path.join(self.DB_PATH, str(class_id))
        
        if not os.path.exists(class_based_path):
            os.makedirs(class_based_path)
            
        filename = f"{safe_name}.jpg"
        file_path = os.path.join(class_based_path, filename)
        
        # Write File
        cv2.imwrite(file_path, img)
        
        # HOT-ADD to memory (Only if this class is currently loaded? No, logic is simpler to just add if loaded)
        # Check if FaceDB is currently serving THIS class? 
        # For simplicity, we just add it. If the DB is on another class, this might be noise, but harmless.
        # Ideally, we should check `face_db.current_class_id`. I'll implement that property.
        
        face_db.add_face_to_memory(safe_name, embedding, class_id)
        
        return {"success": True, "path": file_path, "name": safe_name}


# ==========================================
# CLASS: VIDEO READER
# Threaded camera reading to prevent lag
# ==========================================
class VideoReader:
    def __init__(self, src=0, width=640, height=360):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return self
        try:
            self.cap = cv2.VideoCapture(self.src)
            # Fallback for Mac/USB cams if index fails
            if not self.cap.isOpened() and self.src != 0:
                 print(f"⚠️ Camera {self.src} failed. Trying index 0...")
                 self.cap = cv2.VideoCapture(0)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Try to minimize buffer size (OS independent attempt)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.grabbed, self.frame = self.cap.read()
            self.started = True
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
            print("✅ VideoReader started.")
        except Exception as e:
            print(f"❌ VideoReader failed to start: {e}")
            self.started = False
        return self

    def update(self):
        while self.started:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            # This blocks until a new frame is ready (usually)
            grabbed, frame = self.cap.read()
            
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                # If read fails, wait a bit before retrying
                time.sleep(0.1)

    def read(self):
        with self.read_lock:
            # key: return copy to avoid race conditions during resize/draw
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("🛑 VideoReader stopped.")


# ==========================================
# CLASS: AI CAMERA SYSTEM (MAIN CONTROLLER)
# Integrates Analysis, Database, and Video
# ==========================================
class AICameraSystem:
    def __init__(self):
        # --- MODULES (Instantiated directly now) ---
        self.analyzer = FaceAnalyzer()
        self.face_db = FaceDatabase()
        self.registrar = RegistrationManager()
        
        # --- STANDARD SETUP ---
        self.NUM_WORKERS = 1  # Reduced from 5 to 1 to save CPU/GIL
        self.recognition_queue = queue.Queue(maxsize=10)
        self.known_faces = {}
        self.processing_ids = set()
        self.running = False
        
        # Video & Threading State
        self.frame_lock = threading.Lock()
        self.db_lock = threading.Lock()
        
        # CLEAN FEED SUPPORT
        self.current_clean_jpeg = None
        self.current_jpeg = None
        
        # Async Detection State
        self.latest_frame_for_ai = None
        self.current_boxes = [] 
        self.box_lock = threading.Lock() 
        
        self.active_session_id = None
        self.allow_registration = False # NEW: Toggle for Remote Registration
        
        # Exposed for main.py (Backwards Compatibility)
        self.detector = self.analyzer.validator 
        
        # Initial Load
        # self.reload_database() (Warning: Need to specify class_id now. Start empty.)
        self.active_class_id = None
        
        # LAZY START STATE
        self.threads_started = False
        self.last_accessed = time.time()  # Track activity for Auto-Stop

    def load_class(self, class_id):
        self.active_class_id = class_id
        # Stop worker threads briefly? No, just hot-swap.
        with self.db_lock:
            self.face_db.reload_database(self.analyzer, class_id)
            
    # --- DELEGATED METHODS ---
    def validate_and_stage(self, img_bgr):
        return self.registrar.validate_and_stage(img_bgr, self.analyzer, self.face_db)

    def commit_registration(self, token, name, class_id):
        result = self.registrar.commit_registration(token, name, self.face_db, class_id)
        if result["success"]:
             self.threads_started = False # Why? maybe to reset?
        return result

    # --- THREAD MANAGEMENT ---
    def start_threads(self):
        """Starts the AI (Detection+Recognition) and Display threads."""
        if self.threads_started:
            return
            
        print("🚀 Starting AI & Video Threads...")
        self.running = True
        
        # 1. BRAIN WORKERS (Recognition)
        for i in range(self.NUM_WORKERS):
            t = threading.Thread(target=self._brain_worker, args=(i,), daemon=True)
            t.start()
            
        # 2. DETECTION THREAD (YOLO - Async)
        t_detect = threading.Thread(target=self._detection_loop, daemon=True)
        t_detect.start()
        
        # 3. DISPLAY THREAD (Video - Smooth)
        t_video = threading.Thread(target=self._display_loop, daemon=True)
        t_video.start()
        
        self.threads_started = True

    def stop_loop(self):
        self.running = False
        if hasattr(self, 'reader') and self.reader:
            self.reader.stop()

    def get_latest_frame(self):
        self.last_accessed = time.time()  # Keep Alive
        with self.frame_lock:
            return self.current_jpeg

    def get_clean_frame(self):
        self.last_accessed = time.time()  # Keep Alive
        with self.frame_lock:
            return self.current_clean_jpeg
            
    # --- WORKERS ---
    def _brain_worker(self, worker_id):
        while self.running:
            try:
                try:
                    # UPDATED: Unpack 5 items (Added kpts)
                    track_id, face_crop, proof_frame, bbox, kpts_for_align = self.recognition_queue.get(timeout=1)
                except queue.Empty:
                    continue
                except ValueError:
                    # Handle legacy queue items if any (graceful fallback)
                    continue

                # FIX: Convert BGR to RGB before embedding!
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                target_embedding = self.analyzer.get_embedding(face_rgb, keypoints=kpts_for_align)
                
                name, score_str = self.face_db.get_match(target_embedding)

                # DEBUG: Check Alignment & Norm
                is_aligned_live = "Aligned" if kpts_for_align else "Unaligned"
                norm_val = torch.norm(target_embedding).item()
                # if name == "Unknown":
                # print(f"🔍 Live: {is_aligned_live} | Norm: {norm_val:.4f} | Dist: {score_str}")

                # --- TEMPORAL SMOOTHING (3-Strike Rule) ---
                consecutive_hits = 0
                if track_id is not None:
                    old_data = self.known_faces.get(track_id, {})
                    old_name = old_data.get('name', 'Unknown')
                    old_count = old_data.get('consecutive_count', 0)
                    
                    if name != "Unknown":
                        if name == old_name:
                            consecutive_hits = old_count + 1
                        else:
                            consecutive_hits = 1 # Reset: New Name
                    else:
                        consecutive_hits = 0 # Data lost
                    
                    # Update State
                    self.known_faces[track_id] = {
                        'name': name, 
                        'score': score_str, 
                        'last_seen': time.time(),
                        'consecutive_count': consecutive_hits
                    }

                # --- ATTENDANCE LOGGING ---
                # Only log if we have 3 consecutive hits for the SAME person
                if self.active_session_id and name != "Unknown" and consecutive_hits >= 3:
                    # Fix: Use session-aware lookup to handle duplicate names across classes
                    student = db.get_student_for_session(name, self.active_session_id)
                    if student:
                        # ENROLLMENT CHECK
                        if not db.check_enrollment(student['id'], self.active_session_id):
                             print(f"⚠️ Student {name} recognized but NOT enrolled. Ignoring.")
                        else:
                            with self.db_lock:
                                # SAVE EVIDENCE
                                proof_rel_path = None
                                try:
                                    # 1. Draw Box on Proof Frame
                                    if proof_frame is not None and bbox:
                                        bx1, by1, bx2, by2 = bbox
                                        # Draw Green Box
                                        cv2.rectangle(proof_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                                        # Draw Name Label
                                        cv2.putText(proof_frame, f"{name} {score_str}", (bx1, by1 - 10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        # 2. Prepare Directory: proofs/{session_id}
                                        proof_dir = os.path.join("proofs", str(self.active_session_id))
                                        if not os.path.exists(proof_dir):
                                            os.makedirs(proof_dir)
                                            
                                        # 3. Save File: {student_id}_{timestamp}.jpg (Timestamp ensures uniqueness for raw log)
                                        timestamp = int(time.time()*1000) # Milliseconds for safety
                                        filename = f"{student['id']}_{timestamp}.jpg"
                                        save_path = os.path.join(proof_dir, filename)
                                        
                                        cv2.imwrite(save_path, proof_frame)
                                        proof_rel_path = f"/proofs/{self.active_session_id}/{filename}"
                                        
                                except Exception as e:
                                    print(f"⚠️ Failed to save proof for {name}: {e}")

                                # 1. ALWAYS LOG RAW DATA (User Requirement)
                                db.log_raw_face(self.active_session_id, student['id'], 0.0, proof_rel_path)

                                # 2. LOG OFFICIAL ATTENDANCE (Once per session)
                                db.log_attendance(self.active_session_id, student['id'], "PRESENT", proof_rel_path)

                if track_id in self.processing_ids:
                    self.processing_ids.remove(track_id)

            except Exception as e:
                print(f"Worker {worker_id} Error: {e}")
                if track_id in self.processing_ids:
                    self.processing_ids.remove(track_id)

    def _detection_loop(self):
        """
        AI THREAD: Runs YOLO with Safe Fallback (MPS -> CPU) and NO throttling.
        """
        print("🧠 AI Detection Loop Started (Speed Optimized)")
        
        # FPS Counters
        last_print_time = time.time()
        frame_count = 0

        while self.running:
            try:
                # 1. Get latest frame (Non-destructive)
                if not hasattr(self, 'reader'):
                    time.sleep(0.1)
                    continue
                
                # --- FPS CAP (30 FPS) ---
                start_time = time.time()
                
                ret, frame = self.reader.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Flip once here for consistency
                frame = cv2.flip(frame, 1)
                
                h_orig, w_orig = frame.shape[:2]
                
                # --- OPTIMIZATION ---
                # Run YOLO on smaller image (320px width) for speed on CPU
                target_w = 320
                scale = target_w / w_orig
                target_h = int(h_orig * scale)
                
                small_frame = cv2.resize(frame, (target_w, target_h))

                # 2. Detect
                if self.analyzer.tracker is not None:
                    # REVERT TO CPU (Reliable)
                    # We keep the "No Throttle" logic so it runs as fast as the CPU allows.
                    try:
                        results = self.analyzer.tracker.track(small_frame, verbose=False, conf=0.65, 
                                                        persist=True, device='cpu')
                    except Exception as e:
                        print(f"❌ YOLO Error: {e}")
                        results = None

                    if not results:
                        continue
                        
                    display_boxes = []
                    if results and results[0].boxes:
                        # Get keypoints if available
                        keypoints = results[0].keypoints
                        
                        for i, box in enumerate(results[0].boxes):
                            # --- 5-FEATURE CHECK ---
                            if keypoints is not None and keypoints.conf is not None:
                                kpt_conf = keypoints.conf[i] 
                                mean_conf = kpt_conf.mean()
                                if mean_conf < 0.3:
                                    continue
                                
                                # --- BOX VALIDITY CHECK ---
                                # Get coords on SMALL frame
                                sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                                
                                # SCALE BACK UP TO ORIGINAL
                                x1 = int(sx1 / scale)
                                y1 = int(sy1 / scale)
                                x2 = int(sx2 / scale)
                                y2 = int(sy2 / scale)
                                
                                w = x2 - x1
                                h = y2 - y1
                                
                                # 0. Minimum Size Filter (NEW)
                                if w < 80 or h < 80:
                                    continue
                                
                                # 1. Aspect Ratio Filterik
                                if w > 0 and (h / w) > 3.5:
                                    continue
                                    
                                # 2. Geometric: Keypoints Center MUST be strictly inside the box
                                kpts_xy = keypoints.xy[i].cpu().numpy()
                                
                                # kpts_xy are on small frame
                                k_cx_small = kpts_xy[:, 0].mean()
                                k_cy_small = kpts_xy[:, 1].mean()
                                
                                if not (sx1 < k_cx_small < sx2 and sy1 < k_cy_small < sy2):
                                    continue

                                track_id = int(box.id.item()) if box.id is not None else None
                            
                            # Collect Keypoints for Drawing (OPTIONAL: Scale them up)
                            raw_kpts = []
                            if keypoints is not None:
                                kpts_on_small = keypoints.xy[i].cpu().numpy()
                                for kx, ky in kpts_on_small:
                                    raw_kpts.append((int(kx / scale), int(ky / scale)))
                            
                            display_boxes.append((x1, y1, x2, y2, track_id, raw_kpts))
                            
                            # Queue for Recognition
                            if track_id is not None:
                                if track_id not in self.processing_ids:
                                    should_process = True
                                    if track_id in self.known_faces:
                                        d = self.known_faces[track_id]
                                        if time.time() - d['last_seen'] < 0.2:
                                            should_process = False
                                    
                                    if should_process:
                                        # Crop from ORIGINAL High-Res Frame
                                        cy1 = max(0, y1); cy2 = min(h_orig, y2)
                                        cx1 = max(0, x1); cx2 = min(w_orig, x2)
                                        
                                        if cy2 > cy1 and cx2 > cx1:
                                            face_crop = frame[cy1:cy2, cx1:cx2].copy()
                                            self.processing_ids.add(track_id)
                                            try:
                                                # PREPARE KEYPOINTS FOR ALIGNMENT
                                                # Convert keypoints from Small Frame -> Original Frame -> Relative to Crop
                                                kpts_relative = []
                                                if keypoints is not None:
                                                    raw_kpts = keypoints.xy[i].cpu().numpy()
                                                    for (kx, ky) in raw_kpts:
                                                        # 1. Scale to Orig
                                                        orig_x = kx / scale
                                                        orig_y = ky / scale
                                                        # 2. Relative to Crop (cx1, cy1)
                                                        kpts_relative.append((orig_x - cx1, orig_y - cy1))

                                                # Pass Frame + Box + Keypoints for Evidence & Alignment
                                                # (track_id, crop, full_frame, (x1,y1,x2,y2), keypoints)
                                                self.recognition_queue.put_nowait((track_id, face_crop, frame.copy(), (x1,y1,x2,y2), kpts_relative))
                                            except queue.Full:
                                                pass
                    
                    self.current_boxes = display_boxes
            
            # FPS CALCULATION
                frame_count += 1
                if time.time() - last_print_time > 2.0:
                    fps = frame_count / (time.time() - last_print_time)
                    print(f"📊 YOLO (CPU) FPS: {fps:.1f}")
                    frame_count = 0
                    frame_count = 0
                    last_print_time = time.time()
                
                # Enforce FPS Cap (30 FPS = ~0.033s)
                elapsed = time.time() - start_time
                if elapsed < 0.033:
                    time.sleep(0.033 - elapsed)
                    
            except Exception as e:
                print(f"Detection Loop Error: {e}")
                time.sleep(0.1)

    def _display_loop(self):
        """
        DISPLAY THREAD: Reads -> Draws (using latest boxes) -> Encodes.
        Runs at steady 30 FPS.
        """
        CAMERA_INDEX = 1
        self.reader = VideoReader(src=CAMERA_INDEX, width=1280, height=720).start()
        time.sleep(1.0) # Warmup

        # FPS Counters
        last_print_time = time.time()
        frame_count = 0

        while self.running:
            # AUTO-STOP CHECK
            if time.time() - self.last_accessed > 5.0:
                 print("💤 Camera Idle (5s Timeout). Auto-stopping...")
                 self.stop_loop()
                 self.threads_started = False
                 break
            loop_start = time.time()
            try:
                ret, frame = self.reader.read()
                if not ret: 
                    time.sleep(0.01)
                    continue
                
                frame = cv2.flip(frame, 1) # Consistency

                # Update cleanliness
                self.latest_frame_for_ai = frame.copy()

                # --- DRAWING ---
                boxes_to_draw = self.current_boxes 
                
                for item in boxes_to_draw:
                    x1, y1, x2, y2, track_id = item[:5]
                    color = (0, 255, 255)
                    label = ""
                    
                    if track_id in self.known_faces:
                        d = self.known_faces[track_id]
                        if d['name'] != "Unknown":
                            color = (0, 255, 0)
                            label = f"{d['name']} {d['score']}"
                        else:
                            color = (0, 0, 255)
                            label = "Unknown"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw Keypoints
                    if len(item) == 6:
                        _, _, _, _, _, kpts = item
                        if kpts is not None:
                            for (kx, ky) in kpts:
                                if kx > 0 and ky > 0:
                                    cv2.circle(frame, (kx, ky), 2, (0, 255, 255), -1)

                    if label:
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Encode
                ret, clean_jpeg = cv2.imencode('.jpg', self.latest_frame_for_ai, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    with self.frame_lock:
                        self.current_clean_jpeg = clean_jpeg.tobytes()
                
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    with self.frame_lock:
                        self.current_jpeg = jpeg.tobytes()
                        
            except Exception as e:
                print(f"Display Loop Error: {e}")
            
            # Maintain 60 FPS
            elapsed = time.time() - loop_start
            wait = 0.016 - elapsed
            if wait > 0:
                time.sleep(wait)

            # FPS LOGGING
            frame_count += 1
            if time.time() - last_print_time > 2.0:
                 fps = frame_count / (time.time() - last_print_time)
                 print(f"🎥 Video FPS: {fps:.1f}")
                 frame_count = 0
                 last_print_time = time.time()