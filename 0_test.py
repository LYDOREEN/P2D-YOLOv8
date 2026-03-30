from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('runs/train/yolov8-p2-C2f_ODConv_zengqiang32/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/train/yolov8-p2-C2f_ODConv_zengqiang2/weights/best.pt') # 自己训练结束后的模型权重

    model = YOLO('runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-p2-C2f_ODConv_v3/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-p2-C2f_ODConv_v2/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-p2_7.0/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8-C2f_ODConv/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/7.0/yolov8/weights/best.pt')  # 自己训练结束后的模型权重
    #model = YOLO('runs/train/yolov11/weights/best.pt')  # 自己训练结束后的模型权重

    results = model.val(data='datasets/obstacle_7.0_yolo/obstacle_7.0.yaml',#
              split='test',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='yolov8-p2-C2f_ODConv_small',
              workers=0,
              augment=False,
              )
