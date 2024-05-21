# YOLOv8 training with Roboflow

### Dataset

A foot ball competitions

contain with
    612 images for training
    38 images for validation
    13 images for testing

dataset url: https://universe.roboflow.com/jeju-rvqlq/project-iev0r

### Process

Find a dataset from roboflow universe choose Model for YOLOv8 : https://universe.roboflow.com/search?q=model:yolov8

```powershell
pip install roboflow
```

write a python file for download dataset

```python
from roboflow import Roboflow
rf = Roboflow(api_key="ztiYXwaUGHhu3c7v73hI")
project = rf.workspace("jeju-rvqlq").project("project-iev0r")
version = project.version(1)
dataset = version.download("yolov8")
```

write a python file for training

```python
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8s.pt")  # pass any model type
    results = model.train(data=".\\project-1\\data.yaml", epochs=100, imgsz=640)
```

after finishing our YOLOv8 training, we can see the result in ".\project-1\\runs\detect\train"

### Result

![images](https://github.com/mvclab-ntust-course/course4-nashi999/blob/main/HW2/result/results.png)
![images](https://github.com/mvclab-ntust-course/course4-nashi999/blob/main/HW2/result/train_batch0.jpg)
