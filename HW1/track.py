import cv2
from ultralytics import YOLO

import numpy as np
from collections import defaultdict

model = YOLO("runs/detect/train/weights/best.pt")

video_path = "./argoverse.mp4"

track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture(video_path)

width = 1920
height = 1080
output_size = (width,height)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, output_size)

while cap.isOpened():
    
    success, frame = cap.read()
    
    if success:
        
        results = model.track(frame,persist = True,classes = 2)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        annotates_frames = results[0].plot()
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotates_frames, [points], isClosed=False, color=(0, 255, 255), thickness=2)
        

        cv2.imshow("YOLOx8 Tracking", annotates_frames)
        out.write(cv2.resize(annotates_frames,output_size))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()