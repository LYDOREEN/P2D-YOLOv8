from ultralytics import YOLO
import os


def predict_images_in_folder(
        model_path: str,
        input_folder: str,
        output_folder: str,
        conf_threshold: float = 0.25
) -> None:
    """
    使用YOLOv8模型预测文件夹中的所有图像并只保存标签文件

    参数:
        model_path (str): YOLOv8模型路径(.pt文件)
        input_folder (str): 包含输入图像的文件夹路径
        output_folder (str): 保存预测标签的文件夹路径
        conf_threshold (float): 置信度阈值(默认0.25)
    """
    # 加载模型
    model = YOLO(model_path)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {input_folder}")

    # 处理每张图像
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(output_folder, f"{base_name}.txt")

        # 进行预测
        results = model.predict(
            source=input_path,
            conf=conf_threshold,
            save=False,  # 不保存图片
            save_txt=False,  # 不自动保存标签
            imgsz=640,  # 可以根据需要调整
            iou=0.55
        )

        # 保存标签文件
        with open(output_path, 'w') as f:
            for result in results:
                for box in result.boxes:
                    # 获取YOLO格式的坐标 (class x_center y_center width height)
                    box_data = box.data[0].cpu().numpy()
                    class_id = int(box_data[5])
                    x_center = box_data[0] / result.orig_shape[1]
                    y_center = box_data[1] / result.orig_shape[0]
                    width = box_data[2] / result.orig_shape[1]
                    height = box_data[3] / result.orig_shape[0]
                    confidence = box_data[4]

                    # 写入文件: class x_center y_center width height confidence
                    #f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Saved labels for {image_file} to {output_path}")


# 使用示例
if __name__ == "__main__":
    # 模型路径（可以是YOLOv8的预训练模型或自定义训练模型）
    MODEL_PATH = "runs/train/yolov8-p2-C2f_ODConv-BiFPN/weights/best.pt"  # 替换为你的模型路径

    # 输入图像文件夹路径
    INPUT_FOLDER = r"D:\pytorch_learning\dataprocessing\shiyanqu2_512"  # 替换为你的输入文件夹路径

    # 输出文件夹路径
    OUTPUT_FOLDER = r"D:\pytorch_learning\dataprocessing\output_shiyanqu2"  # 替换为你的输出文件夹路径

    # 运行批量预测
    predict_images_in_folder(
        model_path=MODEL_PATH,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        conf_threshold=0.30
    )