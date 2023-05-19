from ultralytics import YOLO

# Create a new YOLO model from scratch

# model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs


if __name__ == '__main__':
    
    model = YOLO('yolov8.yaml')

    results = model.train(data='lol.yaml', epochs=100)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model('fork.jpg')
    
    
    # Export the model to ONNX format
    success = model.export(format='onnx')

    while True:
        try:
            exec(input(">>> "))
        except Exception as e:
            print(e)