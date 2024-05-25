from ultralytics import YOLO

# Caminho para o arquivo de configuração do dataset
dataset_config = 'dataset.yaml'

# Crie um modelo YOLOv8 de sua escolha (e.g., YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x)
model = YOLO('yolov8n')  

# Treinamento do modelo
model.train(data=dataset_config,
            device='cpu',
            epochs=30,
            patience=8,
            batch=-1,  # number of images per batch (-1 for AutoBatch)
            imgsz=640,
            workers=8,
            pretrained=True,
            resume=False,  # resume training from last checkpoint
            single_cls=False,  # Whether all classes will be the same (just one class)
            # project='runs/detect',  # Default = /home/{user}/Documents/ultralytics/runs
            box=7.5,  # More recall, better IoU, less precission, 
            cls=0.5,  # Bbox class better
            dfl=1.5,  # Distribution Focal Loss. Better bbox boundaries
            val=True,
            # Augmentations
            degrees=0.3,
            hsv_s=0.3,
            hsv_v=0.3,
            scale=0.5,
            fliplr=0.5)
            # save_period=10, project='runs/train', name='exp')

# Salvar o modelo treinado
model.save('best.pt')
