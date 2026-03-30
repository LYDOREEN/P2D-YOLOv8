from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yolov8-p2-C2f_ODConv_7.0/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='datasets/obstacle_7.0_yolo/obstacle_7.0.yaml',
              split='val',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8-p2-C2f_ODConv_7.0',
              workers=0,
              )
