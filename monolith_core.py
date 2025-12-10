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
    def __init__(self, model_path='yolov8n-face.pt'):
        self.MODEL_PATH = model_path
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"⚡ Loading FaceNet (PyTorch) on: {self.device}")

        # Load FaceNet512
        self.recognizer = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device)
        self.recognizer.eval()

        print("🧠 Loading YOLO Models...")
        # 1. TRACKER: For high-speed video loop (CPU)
        self.tracker = YOLO(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else YOLO('yolov8n.pt')
        if os.path.exists(self.MODEL_PATH):
            print(f"✅ Success: Loaded Face Model ({self.MODEL_PATH})")
        else:
            print(f"⚠️ WARNING: '{self.MODEL_PATH}' not found! Loaded Standard YOLO (Objects) instead.")
        # 2. VALIDATOR: For quality checks & reloading (Separate instance to avoid locking)
        self.validator = YOLO(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else YOLO('yolov8n.pt')

    def get_embedding(self, face_img):
        """
        Adds black bars to make it square, then embeds.
        """
        try:
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
            face_img = np.float32(face_img) / 255.0
            face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.recognizer(face_tensor)

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
        if blur_var < 200:  # Threshold for "Clear"
            return False, "Image is too blurry. Please hold steady.", None

        # 2. RUN DETECTION (Use VALIDATOR)
        # We are in the API thread here, so we should use VALIDATOR to avoid race with TRACKER in detection thread
        results = self.validator.predict(img_bgr, verbose=False, conf=0.5, device='cpu')
        
        if not results or not results[0].boxes:
            return False, "No face detected.", None
            
        boxes = results[0].boxes
        if len(boxes) > 1:
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
                 if abs(angle) > 15:
                     return False, "Head tilted. Please keep head straight.", None
                     
                 # B. YAW (Turn) - Distance from Nose to Eyes
                 nose = kpts[2]
                 dist_l_nose = np.linalg.norm(nose - le) # Nose to Left Eye
                 dist_r_nose = np.linalg.norm(nose - re) # Nose to Right Eye
                 
                 # Avoid division by zero
                 if dist_l_nose > 0 and dist_r_nose > 0:
                     ratio = max(dist_l_nose, dist_r_nose) / min(dist_l_nose, dist_r_nose)
                     # Frontal face should have ratio close to 1.0. 
                     # Side face will have large ratio.
                     if ratio > 2.0: # Tunable
                         return False, "Face turned. Please look straight at camera.", None

        return True, "Quality OK", face_crop


# ==========================================
# CLASS: FACE DATABASE
# Handles loading gallery, caching, and matching
# ==========================================
class FaceDatabase:
    def __init__(self, db_path="gallery", confidence_threshold=0.81, device=None):
        self.DB_PATH = db_path
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.face_db_embeddings = []
        self.face_db_names = []
        
        # Cache file path
        self.CACHE_PATH = os.path.join(self.DB_PATH, "vectors.pt")

    def reload_database(self, face_analyzer):
        """Loads gallery images with CACHING support."""
        print("📂 Indexing Gallery...")
        
        # 1. Load Cache if exists
        cached_data = {}
        if os.path.exists(self.CACHE_PATH):
            try:
                print("   Loading cache file...")
                cached_data = torch.load(self.CACHE_PATH)
                print(f"   Cache loaded: {len(cached_data)} entries.")
            except Exception as e:
                print(f"⚠️ Corrupt cache, rebuilding: {e}")

        # 2. Scan Files
        gallery_images = glob.glob(os.path.join(self.DB_PATH, "**", "*.jpg"), recursive=True)
        gallery_images += glob.glob(os.path.join(self.DB_PATH, "**", "*.png"), recursive=True)
        
        # 3. Determine work to do
        new_names = []
        new_embeddings = []
        
        valid_cache_count = 0
        processed_count = 0
        
        for img_path in gallery_images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            
            # If in cache, use it!
            if name in cached_data:
                new_names.append(name)
                new_embeddings.append(cached_data[name])
                valid_cache_count += 1
                continue
                
            # Not in cache, Process it
            try:
                processed_count += 1
                img = cv2.imread(img_path)
                if img is None: continue
                
                # OPTIMIZATION: Resize huge images
                h, w = img.shape[:2]
                if w > 640:
                    scale = 640 / w
                    img = cv2.resize(img, (640, int(h * scale)))
                
                # Detect
                results = face_analyzer.validator.predict(img, verbose=False, conf=0.15, device='cpu')
                
                face_crop = None
                if results and results[0].boxes:
                    largest_area = 0
                    best_box = None
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            best_box = (x1, y1, x2, y2)
                    
                    if best_box:
                        x1, y1, x2, y2 = best_box
                        face_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

                if face_crop is None or face_crop.size == 0:
                    print(f"⚠️ No face detected in {name}, using full image.")
                    face_crop = img

                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                embedding = face_analyzer.get_embedding(face_crop)
                
                if embedding is not None:
                    # Move to CPU for storage in list/cache (save GPU memory)
                    emb_cpu = embedding.cpu()
                    new_embeddings.append(emb_cpu)
                    new_names.append(name)
                    
                    # Update cache dict immediately
                    cached_data[name] = emb_cpu
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

        # 4. Save Cache (if we did any work)
        if processed_count > 0:
            print(f"💾 Saving updated cache with {len(cached_data)} entries...")
            torch.save(cached_data, self.CACHE_PATH)
        
        # 5. Build Final Tensor
        if new_embeddings:
            # Stack and move to Device (MPS) ONCE
            self.face_db_embeddings = torch.stack(new_embeddings).to(self.device)
            self.face_db_names = new_names
            print(f"✅ Reload Complete. Active Faces: {len(self.face_db_names)} (Cached: {valid_cache_count}, New: {processed_count})")
        else:
            self.face_db_embeddings = []
            self.face_db_names = []
            print("⚠️ No faces found.")

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
        
        return name, score_str
        
    def find_nearest_dist(self, target_embedding):
        """Returns the distance to the nearest face in DB."""
        if target_embedding is None or len(self.face_db_embeddings) == 0:
            return float('inf'), None
            
        diff = self.face_db_embeddings - target_embedding
        dist = torch.norm(diff, dim=1).cpu().numpy()
        min_idx = np.argmin(dist)
        return dist[min_idx], self.face_db_names[min_idx]

    def add_face_to_memory(self, name, embedding):
        """
        Directly appends a new face to the running memory & updates cache.
        """
        if embedding is None: return

        print(f"⚡ Hot-adding {name} to memory...")
        
        # 1. Update In-Memory Tensor
        emb_tensor = embedding.to(self.device)
        if len(self.face_db_embeddings) == 0:
            self.face_db_embeddings = emb_tensor.unsqueeze(0)
        else:
            self.face_db_embeddings = torch.cat((self.face_db_embeddings, emb_tensor.unsqueeze(0)))
            
        self.face_db_names.append(name)
        
        # 2. Update Disk Cache (So next restart is fast)
        try:
            if os.path.exists(self.CACHE_PATH):
                cached_data = torch.load(self.CACHE_PATH)
            else:
                cached_data = {}
            
            cached_data[name] = embedding.cpu()
            torch.save(cached_data, self.CACHE_PATH)
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

    def commit_registration(self, token, name, face_db):
        """
        Moves from cache to gallery. Returns file path.
        """
        if token not in self.registration_cache:
            return {"success": False, "message": "Session expired or invalid token."}
            
        data = self.registration_cache.pop(token)
        img = data['full_img']
        embedding = data['embedding']
        
        # Save to Gallery
        safe_name = "".join(x for x in name if x.isalnum() or x in " _-").strip()
        if not os.path.exists(self.DB_PATH):
            os.makedirs(self.DB_PATH)
            
        filename = f"{safe_name}.jpg"
        file_path = os.path.join(self.DB_PATH, filename)
        
        # Write File
        cv2.imwrite(file_path, img)
        
        # HOT-ADD to memory
        face_db.add_face_to_memory(safe_name, embedding)
        
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
        
        # Exposed for main.py (Backwards Compatibility)
        self.detector = self.analyzer.validator 
        
        # Initial Load
        self.reload_database()
        
        # LAZY START STATE
        self.threads_started = False

    def reload_database(self):
        self.face_db.reload_database(self.analyzer)

    # --- DELEGATED METHODS ---
    def validate_and_stage(self, img_bgr):
        return self.registrar.validate_and_stage(img_bgr, self.analyzer, self.face_db)

    def commit_registration(self, token, name):
        result = self.registrar.commit_registration(token, name, self.face_db)
        if result["success"]:
             self.threads_started = False
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
        with self.frame_lock:
            return self.current_jpeg

    def get_clean_frame(self):
        with self.frame_lock:
            return self.current_clean_jpeg
            
    # --- WORKERS ---
    def _brain_worker(self, worker_id):
        while self.running:
            try:
                try:
                    track_id, face_crop = self.recognition_queue.get(timeout=1)
                except queue.Empty:
                    continue

                target_embedding = self.analyzer.get_embedding(face_crop)
                name, score_str = self.face_db.get_match(target_embedding)

                if track_id is not None:
                    # LABEL SMOOTHING logic
                    update = True
                    if track_id in self.known_faces:
                        old_data = self.known_faces[track_id]
                        if old_data['name'] != "Unknown" and name == "Unknown":
                            if time.time() - old_data['last_seen'] < 2.0: # Keep known name for 2s
                                update = False

                    if update:
                        self.known_faces[track_id] = {'name': name, 'score': score_str, 'last_seen': time.time()}

                if self.active_session_id and name != "Unknown":
                    student = db.get_student_by_name(name)
                    if student:
                        with self.db_lock:
                            db.log_attendance(self.active_session_id, student['id'])

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
                
                # REMOVED THROTTLE: Run as fast as possible
                
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
                                
                                # 1. Aspect Ratio Filter
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
                                        if time.time() - d['last_seen'] < 1.0:
                                            should_process = False
                                    
                                    if should_process:
                                        # Crop from ORIGINAL High-Res Frame
                                        cy1 = max(0, y1); cy2 = min(h_orig, y2)
                                        cx1 = max(0, x1); cx2 = min(w_orig, x2)
                                        
                                        if cy2 > cy1 and cx2 > cx1:
                                            face_crop = frame[cy1:cy2, cx1:cx2].copy()
                                            self.processing_ids.add(track_id)
                                            try:
                                                self.recognition_queue.put_nowait((track_id, face_crop))
                                            except queue.Full:
                                                pass
                    
                    self.current_boxes = display_boxes
            
            # FPS CALCULATION
                frame_count += 1
                if time.time() - last_print_time > 2.0:
                    fps = frame_count / (time.time() - last_print_time)
                    print(f"📊 YOLO (CPU) FPS: {fps:.1f}")
                    frame_count = 0
                    last_print_time = time.time()
                    
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