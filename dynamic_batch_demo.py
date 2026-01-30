
"""
CONCEPTUAL DEMO: Dynamic Batching Architecture
This shows how we would rewrite the "Brain" to handle 10 faces at once.

Current Architecture (Threaded/Serial):
----------------------------------------
Face 1 -> [Worker Thread 1] -> [GPU Request] -> Result
Face 2 -> [Worker Thread 2] -> [GPU Request] -> Result
...
Face 10 -> [Worker Thread 4 (Wait for 1 to finish)] -> [GPU Request] -> Result

Problem: The GPU is interrupted 10 times. Overhead is high. "Queue" fills up.


Dynamic Batching Architecture:
----------------------------------------
Face 1, Face 2, ... Face 10 -> [Batch Collector] 
       |
  (Wait 30ms or until 8 faces collected)
       |
  [ Tensor: Shape(10, 3, 160, 160) ]  <-- One Giant Block
       |
  [GPU Request (ONE CALL)] 
       |
  [ Results: 10 Embeddings ]
       |
  [Distributor] -> Face 1 Result, Face 2 Result...

Why it is "Hard":
1. We need a "Collector Loop" that sits between detection and recognition.
2. We need detailed timeout logic (what if only 1 person? Don't wait forever).
3. We need to match the results back to the original Track IDs (bookkeeping).
"""

import time
import torch
import numpy as np
import threading
import queue

class DynamicBatcher:
    def __init__(self, model, batch_size=8, timeout=0.05):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        
        # Introduction Queue (Faces enter here)
        self.input_queue = queue.Queue()
        
        # Results Dictionary (To send answers back)
        self.results = {} 
        self.results_events = {} # Events to wake up waiting threads
        
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_loop)
        self.batch_thread.start()

    def predict(self, face_tensor, track_id):
        """
        Called by the Worker.
        Instead of running model() directly, it submits to the batcher 
        and sleeps until the batch is done.
        """
        # 1. Create a "Ticket" (Event)
        event = threading.Event()
        self.results_events[track_id] = event
        
        # 2. Submit to Queue
        self.input_queue.put((track_id, face_tensor))
        
        # 3. SLEEP (Wait for the batch bus to leave)
        event.wait() 
        
        # 4. Wake up and return result
        result = self.results.pop(track_id)
        return result

    def _batch_loop(self):
        print("🚌 Batch Bus Started...")
        while self.running:
            batch_items = []
            
            # --- COLLECT PHASE ---
            start_wait = time.time()
            while len(batch_items) < self.batch_size:
                # Calculate time left
                time_left = self.timeout - (time.time() - start_wait)
                
                if time_left <= 0 and len(batch_items) > 0:
                    break # Timeout! Go with what we have.
                
                try:
                    # Wait for passengers
                    item = self.input_queue.get(timeout=max(0.001, time_left))
                    batch_items.append(item)
                except queue.Empty:
                    if len(batch_items) > 0:
                        break # Timeout!
                    else:
                        continue # Keep waiting if empty
            
            if not batch_items:
                continue

            # --- PROCESS PHASE (The "One GPU Call") ---
            # 1. Stack all tensors into one giant block
            # List of (track_id, tensor)
            tensors = [x[1] for x in batch_items] 
            ids = [x[0] for x in batch_items]
            
            # Stack: [1, 3, 160, 160] -> [N, 3, 160, 160]
            batch_tensor = torch.cat(tensors, dim=0) 
            
            # 2. RUN MODEL (ONCE!)
            print(f"🚀 Running Batch of {len(batch_tensor)} faces...")
            with torch.no_grad():
                # This call takes roughly the same time for 1 face or 8 faces on a GPU
                embeddings = self.model(batch_tensor) 
            
            # 3. DISTRIBUTE RESULTS
            for i, track_id in enumerate(ids):
                self.results[track_id] = embeddings[i]
                # Wake up the waiting thread
                self.results_events[track_id].set()

# ==========================================
# MOCK USAGE
# ==========================================
if __name__ == "__main__":
    # Mock FaceNet Model (Just identity)
    def mock_facenet(tensors):
        time.sleep(0.1) # Simulate GPU latency
        return tensors[:, 0, 0, 0] # Dummy output

    batcher = DynamicBatcher(mock_facenet)

    # Simulate 10 Threads asking at once
    def worker(i):
        # Fake Face Tensor
        face = torch.zeros(1, 3, 160, 160)
        print(f"👤 Face {i} waiting...")
        result = batcher.predict(face, i)
        print(f"✅ Face {i} Done!")

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
        
    batcher.running = False
