import cv2
import time
import threading
import queue
import database as db
import os

# --- NEW MODULES ---
from face_analysis import FaceAnalyzer
from face_database import FaceDatabase
from registration_manager import RegistrationManager


class VideoReader:
    """
    Dedicated thread for reading frames from the camera.
    This ensures that the main processing loop always gets the *latest* frame,
    preventing buffer buildup and latency (lag).
    """
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

class AICameraSystem:
    def __init__(self):
        # --- MODULES ---
        self.analyzer = FaceAnalyzer()
        self.face_db = FaceDatabase()
        self.registrar = RegistrationManager()
        
        # --- STANDARD SETUP ---
        self.NUM_WORKERS = 5
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
        self.box_lock = threading.Lock() # Still used for atomic read/write locally
        
        self.active_session_id = None
        
        # Exposed for main.py (Backwards Compatibility)
        self.detector = self.analyzer.validator # For main.py direct access if needed
        
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
             # If we wanted to trigger a full reload, we could, but hot-add is better.
             # self.threads_started = False # Forces restart if needed (Optional)
             # Actually, let's keep the existing logic of core_AI.py which reset threads:
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

                # Smoother Console Log
                # print(f"⚡ Worker {worker_id}: {name} {score_str}")

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
        AI THREAD: Runs YOLO as fast as possible (or MPS limited).
        Updates self.current_boxes.
        """
        print("🧠 AI Detection Loop Started (MPS Enabled)")
        while self.running:
            try:
                # 1. Get latest frame (Non-destructive)
                if not hasattr(self, 'reader'):
                    time.sleep(0.1)
                    continue
                    
                ret, frame = self.reader.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Flip once here for consistency if needed, but Reader usually gives raw.
                # Let's assume Reader gives raw, we flip in Display. 
                # Actually, consistency matters. Let's flip in Reader or consistently everywhere.
                # Current code flipped in _video_loop.
                frame = cv2.flip(frame, 1)

                # 2. Detect
                if self.analyzer.tracker is not None:
                    # Run at Full Resolution (640p) for better accuracy against background noise
                    # Increased confidence to 0.65 to reduce ghost boxes
                    results = self.analyzer.tracker.track(frame, verbose=False, conf=0.65, 
                                                    persist=True, device='mps')
                    
                    display_boxes = []
                    if results and results[0].boxes:
                        # Get keypoints if available
                        keypoints = results[0].keypoints
                        
                        for i, box in enumerate(results[0].boxes):
                            # --- 5-FEATURE CHECK ---
                            # Check if this specific box has valid keypoints
                            if keypoints is not None and keypoints.conf is not None:
                                # Get confidence scores for the 5 points of this face
                                kpt_conf = keypoints.conf[i] 
                                mean_conf = kpt_conf.mean()
                                if mean_conf < 0.3:
                                    print(f"🚫 Ignored 'face' (Staircase?) due to low keypoint confidence: {mean_conf:.2f}")
                                    continue
                                
                                # --- BOX VALIDITY CHECK ---
                                _x1, _y1, _x2, _y2 = map(int, box.xyxy[0])
                                x1, x2 = min(_x1, _x2), max(_x1, _x2)
                                y1, y2 = min(_y1, _y2), max(_y1, _y2)
                                w = x2 - x1
                                h = y2 - y1
                                
                                # 1. Aspect Ratio Filter (Staircase Pillars are usually strips)
                                # Faces are usually 1:1 to 1:1.5. A ratio > 3.5 is definitely not a face.
                                if w > 0 and (h / w) > 3.5:
                                    print(f"🚫 Ignored 'Strip' Box: Ratio {h/w:.1f} too high. {w}x{h}")
                                    continue
                                    
                                # 2. Geometric: Keypoints Center MUST be strictly inside the box
                                # Centroid of keypoints
                                kpts_xy = keypoints.xy[i].cpu().numpy()
                                k_cx = kpts_xy[:, 0].mean()
                                k_cy = kpts_xy[:, 1].mean()
                                
                                # Strict containment (no margin) because ghosts are often NEAR the face logic
                                if not (x1 < k_cx < x2 and y1 < k_cy < y2):
                                    print(f"🚫 Ignored 'Ghost': Keypoints outside box. Center=({k_cx:.0f},{k_cy:.0f}) Box=[{x1},{y1},{x2},{y2}]")
                                    continue
                                # -----------------------

                                print(f"✅ Approved Face: {mean_conf:.2f} Size: {w}x{h}")
                                
                                track_id = int(box.id.item()) if box.id is not None else None
                            
                            # Collect Keypoints for Drawing
                            raw_kpts = keypoints.xy[i].cpu().numpy().astype(int) if keypoints is not None else None
                            
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
                                        h, w = frame.shape[:2]
                                        cy1 = max(0, y1); cy2 = min(h, y2)
                                        cx1 = max(0, x1); cx2 = min(w, x2)
                                        
                                        if cy2 > cy1 and cx2 > cx1:
                                            face_crop = frame[cy1:cy2, cx1:cx2].copy()
                                            self.processing_ids.add(track_id)
                                            try:
                                                self.recognition_queue.put_nowait((track_id, face_crop))
                                            except queue.Full:
                                                pass
                    
                    # Atomic update
                    self.current_boxes = display_boxes
                    
            except Exception as e:
                print(f"Detection Loop Error: {e}")
                time.sleep(0.1)

    def _display_loop(self):
        """
        DISPLAY THREAD: Reads -> Draws (using latest boxes) -> Encodes.
        Runs at steady 30 FPS.
        """
        CAMERA_INDEX = 1
        CAMERA_INDEX = 1
        self.reader = VideoReader(src=CAMERA_INDEX, width=1280, height=720).start()
        time.sleep(1.0) # Warmup

        while self.running:
            loop_start = time.time()
            try:
                ret, frame = self.reader.read()
                if not ret: 
                    time.sleep(0.01)
                    continue
                
                frame = cv2.flip(frame, 1) # Consistency

                # Update cleanliness
                self.latest_frame_for_ai = frame.copy() # Actually unused by AI loop now (it reads directly), but nice to have.

                # --- DRAWING ---
                # Use whatever boxes we have (even if 1 frame old)
                # This guarantees smooth video even if AI is 20 FPS.
                boxes_to_draw = self.current_boxes 
                
                for item in boxes_to_draw:
                    # Unpack flexible tuple (updates might change size)
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
                    
                    # Draw Keypoints (from tuple: x1, y1, x2, y2, track_id, kpts)
                    # Unpack carefully - we added kpts to the tuple!
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
            
            # Maintain ~30 FPS
            elapsed = time.time() - loop_start
            wait = 0.033 - elapsed
            if wait > 0:
                time.sleep(wait)