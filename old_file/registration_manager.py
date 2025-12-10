import time
import uuid
import base64
import cv2
import os

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
