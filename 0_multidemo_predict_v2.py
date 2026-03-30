from ultralytics import YOLO
import os


def predict_images_in_folder(
        model_path: str,
        input_folder: str,
        output_folder: str,
        # conf_threshold: float = 0.25
) -> None:
    """
    使用YOLOv8模型预测文件夹中的所有图像并保存标签文件（简化版）

    参数:
        model_path: YOLOv8模型路径(.pt文件)
        input_folder: 输入图像文件夹路径
        output_folder: 输出标签文件夹路径
        conf_threshold: 置信度阈值(默认0.25)
    """
    # 加载模型
    model = YOLO(model_path)

    # 进行预测并自动保存标签
    results = model.predict(
        source=input_folder,
        # conf=conf_threshold,
        save=False,  # 不保存可视化图片
        save_txt=True,  # 自动保存标签文件
        save_conf=True,  # 不在标签中包含置信度
        project=output_folder,  # 指定输出目录
        name='',  # 空名字使文件直接保存在output_folder
        exist_ok=True,  # 允许覆盖已有文件
        # iou = 0.6
    )


# 使用示例
if __name__ == "__main__":
    predict_images_in_folder(
        #model_path="runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt",
        #model_path="runs/7.0/yolov8-p2-C2f_ODConv_v3/weights/best.pt",
        #model_path="runs/7.0/yolov8-p2-C2f_ODConv_v2/weights/best.pt",
        #model_path="runs/7.0/yolov8-p2_7.0/weights/best.pt",
        #model_path="runs/7.0/yolov8-C2f_ODConv/weights/best.pt",
        #model_path="runs/7.0/yolov8/weights/best.pt",
        #model_path="runs/train/yolov5/weights/best.pt",
        #model_path="runs/train/yolov11/weights/best.pt",
        #model_path="runs/7.0/yolov8/weights/best.pt",
        model_path="runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt",
        # model_path="runs/train/yolov8-p2-C2f_ODConv_zengqiang2/weights/best.pt",
        input_folder=r"D:\pytorch_learning\dataprocessing\keshihuajiance4\images",
        output_folder=r"D:\pytorch_learning\ultralytics-main\output_6",
    )