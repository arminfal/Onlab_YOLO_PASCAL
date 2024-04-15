#Training
from ultralytics import YOLO

# Load a model
model = YOLO('model.pt')  # load a pretrained model
#model = YOLO('G:\\Onlab\\VOC_dataset\\VOCdevkit\\VOC2012\\runs\\segment\\train\\weights\\last.pt')# load the last training model
# Train the model with the tuned hyperparameters
results = model.train(data='dataset.yaml', epochs=5, imgsz=500, batch=-1, workers=8, optimizer='AdamW', patience=30, lr0= 0.0102, lrf= 0.00387, momentum= 0.81859, weight_decay= 0.00068, warmup_epochs= 3.46434, warmup_momentum= 0.89747, box= 8.86819, cls= 0.80673, dfl= 1.31027, hsv_h= 0.01541, hsv_s= 0.74576, hsv_v= 0.3294, degrees= 0.0, translate= 0.10281, scale= 0.19924, shear= 0.0, perspective= 0.0, flipud= 0.0, fliplr= 0.39194, mosaic= 0.68242, mixup= 0.0, copy_paste= 0.0)