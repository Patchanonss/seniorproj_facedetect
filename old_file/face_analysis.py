import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

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
