from ultralytics import YOLO

def train_yolo(version,save_path,log_file):

    model = YOLO(version)

    results = model.train(
        data="/home/andreevaleksandr/Documents/YOLO/Drone_dataset_main/data.yaml",
        epochs=1,
        patience = 10,
        batch=-1,
        imgsz=320,
        save_period = 1,
        project=save_path,
        plots = True,
        device = 0,
        save_json=True
    )   

    with open(log_file,'a') as log:
        log.write(f'Training for version{version} completed.\n')
        log.write(str(results)+ "\n")

if __name__ =="__main__":
    training_jobs = [
        {"version":"yolov8n","save_path":"results/yolov8n","log_file":"training_yolov8n.log"},
        {"version":"yolo11n","save_path":"results/yolo11n","log_file":"training_yolo11n.log"},
        {"version":"yolov8m","save_path":"results/yolov8m","log_file":"training_yolov8m.log"}
    ]

    for job in training_jobs:
        print(f"Starting training for {job['version']}...")
        train_yolo(
            job["version"],
            job["save_path"],
            job["log_file"]
        )
        print (f"Training for {job['version']} completed.")

    print ("All trainings completed.")