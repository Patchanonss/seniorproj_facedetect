
import cv2
import numpy as np
import os
import glob
import torch
import random
from monolith_core import FaceAnalyzer, FaceDatabase, AICameraSystem

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * 50
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def run_stress_test(class_id):
    print(f"🔥 Starting Synthetic Stress Test for Class {class_id}...")
    
    # 1. Initialize System
    print("   -> Initializing AI System...")
    sys = AICameraSystem()
    sys.face_db.reload_database(sys.analyzer, class_id)
    
    if len(sys.face_db.face_db_names) == 0:
        print("❌ No faces in database. Register students first.")
        return

    # 2. Load Gallery Images directly (Ground Truth)
    gallery_path = f"gallery/{class_id}"
    images = glob.glob(os.path.join(gallery_path, "*.jpg")) + glob.glob(os.path.join(gallery_path, "*.png"))
    
    total_tests = 0
    passes = 0
    
    print(f"   -> Found {len(images)} ground truth faces.")
    
    for img_path in images:
        true_name = os.path.splitext(os.path.basename(img_path))[0]
        original_img = cv2.imread(img_path)
        
        if original_img is None: continue
        
        # GENERATE TEST CASES
        test_cases = []
        
        # Case 1: Brightness Low
        test_cases.append(("Darkened", adjust_brightness(original_img, 0.5)))
        
        # Case 2: Brightness High
        test_cases.append(("Brightened", adjust_brightness(original_img, 1.5)))
        
        # Case 3: Rotation +15 deg (Simulate Head Tilt)
        test_cases.append(("Rotated +15", rotate_image(original_img, 15)))
        
        # Case 4: Rotation -15 deg
        test_cases.append(("Rotated -15", rotate_image(original_img, -15)))
        
        # Case 5: Noise
        test_cases.append(("Noisy", add_noise(original_img)))
        
        # Case 6: Scaling (Simulate Distance)
        h, w = original_img.shape[:2]
        small = cv2.resize(original_img, (int(w*0.5), int(h*0.5)))
        test_cases.append(("Small (50%)", small))

        print(f"\n🧪 Testing Subject: {true_name}")
        
        for case_name, test_img in test_cases:
            total_tests += 1
            
            # --- RUN RECOGNITION PIPELINE ---
            # We skip the tracker and go straight to embedding extraction to isolate recognition logic
            # mimicking the behavior of _brain_worker
            
            # 1. Detect (simulate tracker finding the face)
            results = sys.analyzer.validator.predict(test_img, verbose=False, conf=0.5)
            
            if not results or not results[0].boxes:
                print(f"   ❌ {case_name}: Detection Failed")
                continue
                
            # Assume largest face is target
            box = results[0].boxes[0] # Simply take first
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            face_crop = test_img[max(0, y1):min(test_img.shape[0], y2), max(0, x1):min(test_img.shape[1], x2)]
            
            # Get Keypoints
            kpts_relative = None
            if results[0].keypoints is not None:
                raw_kpts = results[0].keypoints.xy[0].cpu().numpy()
                kpts_relative = []
                for (kx, ky) in raw_kpts:
                    kpts_relative.append((kx - x1, ky - y1))
            
            # 2. Get Embedding
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            embedding = sys.analyzer.get_embedding(face_rgb, keypoints=kpts_relative)
            
            if embedding is None:
                print(f"   ❌ {case_name}: Alignment Failed")
                continue
                
            # 3. Match
            pred_name, score = sys.face_db.get_match(embedding)
            
            if pred_name == true_name:
                print(f"   ✅ {case_name}: Matched {score}")
                passes += 1
            else:
                print(f"   ⚠️ {case_name}: Wrong Match! Got {pred_name} {score}")

    print(f"\n📊 RESULTS: {passes}/{total_tests} passed ({(passes/total_tests)*100:.1f}%)")

if __name__ == "__main__":
    # Default to class 6 (User's class) if not specified
    import sys
    cid = sys.argv[1] if len(sys.argv) > 1 else "6"
    run_stress_test(cid)
