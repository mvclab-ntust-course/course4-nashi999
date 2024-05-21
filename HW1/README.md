# YOLOv8

trained with argoverse dataset

### dataset

download dataset url : https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link

### train.yaml

For training and convert dataset from Argoverse annotations to YOLO labels

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Argoverse-HD dataset (ring-front-center camera) https://www.cs.cmu.edu/~mengtial/proj/streaming/ by Argo AI
# Documentation: https://docs.ultralytics.com/datasets/detect/argoverse/
# Example usage: yolo train data=Argoverse.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── Argoverse  ← downloads here (31.5 GB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./datasets/Argoverse/Argoverse-HD-Full # dataset root dir
train: Argoverse-1.1/images/train/ # train images (relative to 'path') 39384 images
val: Argoverse-1.1/images/val/ # val images (relative to 'path') 15062 images

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
  6: traffic_light
  7: stop_sign

# Download script/URL (optional) ---------------------------------------------------------------------------------------
download: |
  import json
  from tqdm import tqdm
  from ultralytics.utils.downloads import download
  from pathlib import Path

  def argoverse2yolo(set):
      labels = {}
      a = json.load(open(set, "rb"))
      for annot in tqdm(a['annotations'], desc=f"Converting {set} to YOLOv5 format..."):
          img_id = annot['image_id']
          img_name = a['images'][img_id]['name']
          img_label_name = f'{img_name[:-3]}txt'

          cls = annot['category_id']  # instance class id
          x_center, y_center, width, height = annot['bbox']
          x_center = (x_center + width / 2) / 1920.0  # offset and scale
          y_center = (y_center + height / 2) / 1200.0  # offset and scale
          width /= 1920.0  # scale
          height /= 1200.0  # scale

          img_dir = set.parents[2] / 'Argoverse-1.1' / 'labels' / a['seq_dirs'][a['images'][annot['image_id']]['sid']]
          if not img_dir.exists():
              img_dir.mkdir(parents=True, exist_ok=True)

          k = str(img_dir / img_label_name)
          if k not in labels:
              labels[k] = []
          labels[k].append(f"{cls} {x_center} {y_center} {width} {height}\n")

      for k in labels:
          with open(k, "w") as f:
              f.writelines(labels[k])


  # Download 'https://argoverse-hd.s3.us-east-2.amazonaws.com/Argoverse-HD-Full.zip' (deprecated S3 link)
  path = str(Path('./datasets/Argoverse/Argoverse-HD-Full'))
  urls = ['https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link']
  dir = Path(path)  # dataset root dir

  # Convert
  annotations_dir = 'Argoverse-HD/annotations/'
  (dir / 'Argoverse-1.1' / 'tracking').rename(dir / 'Argoverse-1.1' / 'images')  # rename 'tracking' to 'images'
  for d in "train.json", "val.json":
      argoverse2yolo(dir / annotations_dir / d)  # convert Argoverse annotations to YOLO labels
```


### training code

```python
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # pass any model type
    results = model.train(data='train.yaml', epochs=1, imgsz=640)
```

### tracking code

for loop at line 35 is for drawing track

cv2.VideoWriter is for saving result video

```python
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
```
