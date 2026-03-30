#from ultralytics.nn.modules.head import Detect_DyHead
from ultralytics import YOLO
#from ultralytics import RTDETR
def train_yolov8():
    # 加载预训练模型
    model = YOLO("yolov8-BiFPN.yaml")  # 使用 YOLOv11n 模型

    # 训练模型
    results = model.train(
        data="datasets/obstacle_3.0/obstacle_3.0.yaml",  # 数据集配置文件
        epochs=200,                 # 训练轮数
        batch=16,                   # 批量大小
        imgsz=640,                  # 图像尺寸
        device="0",                 # 使用 GPU（"0" 表示第一块 GPU）
        name="yolov8_train_20250326_",       # 训练任务名称
        project='runs/train',
        workers=0,
        save_period=10,

        # resume=True
    )

    # 打印训练结果
    print(results)

if __name__ == "__main__":
    train_yolov8()