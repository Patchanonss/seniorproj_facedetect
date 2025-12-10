import os
import glob
import cv2
import torch
import numpy as np
import time

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
        # We check modification times could be risky if copied.
        # Simple check: do we have this filename in cache?
        
        new_names = []
        new_embeddings = []
        
        valid_cache_count = 0
        processed_count = 0
        
        for img_path in gallery_images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            
            # If in cache, use it!
            if name in cached_data:
                # Optional: Check file mtime vs cache timestamp if we stored it
                # For now, assume filename unique per person = valid
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
            # Optimized: Keep persistent db embeddings on GPU
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
