from ultralytics import YOLO

def start_training():
    # Load a small, fast model (YOLOv8 Nano) - best for game bots
    model = YOLO('yolov8n.pt') 

    # Start training
    results = model.train(
        data=r'C:\Users\Vince\Desktop\arch2\data\yolo_dataset\data.yaml',
        epochs=50,         # Start with 50; you can increase later
        imgsz=640,         # Standard YOLO resolution
        batch=16,          # Adjust based on your GPU VRAM (8, 16, or 32)
        device=0,          # Use GPU (0). Use 'cpu' if no NVIDIA GPU
        name='SenseNova_v1'
    )

if __name__ == "__main__":
    start_training()