#from ultralytics.nn.modules.head import Detect_DyHead
from ultralytics import YOLO
#from ultralytics import RTDETR
def train_yolov8():
    # 加载预训练模型
    model = YOLO("yolov8-p2-C2f_ODConv.yaml")  # 使用 YOLOv11n 模型

    # 训练模型
    results = model.train(
        data="datasets/obstacle_7.0_yolo/obstacle_7.0.yaml",  # 数据集配置文件
        epochs=200,                 # 训练轮数
        batch=16,                   # 批量大小
        imgsz=640,                  # 图像尺寸
        device="0",                 # 使用 GPU（"0" 表示第一块 GPU）
        name="yolov8-p2-C2f_ODConv_7.0",       # 训练任务名称
        project='runs/train',
        workers=0,
        save_period=10,
        #resume=True,
    )


    # 打印训练结果
    print(results)

if __name__ == "__main__":
    train_yolov8()