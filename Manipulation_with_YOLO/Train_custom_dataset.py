import os
from comet_ml import Experiment
from ultralytics import YOLO

# experiment = Experiment(
# api_key = 'Ylh7DPsTIoPSqr7aQzGuxMeH4',
# project_name="yolo-comparison",
# workspace="anonymous210370"
# )

model = YOLO("yolo11n.pt")

results = model.train(
    data="/home/andreevaleksandr/Documents/YOLO/Drone_dataset_main/data.yaml",
    epochs=50,
    cache = True,
    single_cls = True,
    patience = 10,
    batch=16,
    imgsz=640,
    save_period = 5,
    project="comet-example-yolov11n",
    plots = True,
    device = 0,
    save_json=True
)