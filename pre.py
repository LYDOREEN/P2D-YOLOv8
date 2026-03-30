from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os


def predict_single_image(
        image_path: str,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
) -> None:
    """
    使用YOLOv8模型对单张图片进行目标检测，并显示结果

    参数:
        image_path (str): 图片路径
        model_path (str): 训练好的YOLOv8模型路径(.pt)
        conf_threshold (float): 置信度阈值
        iou_threshold (float): NMS的IOU阈值
    """
    # 检查图片是否存在
    if not os.path.isfile(image_path):
        print(f"错误: 找不到图片文件 {image_path}")
        return

    # 加载模型
    model = YOLO(model_path)

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    # 进行预测
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        save=False  # 不保存图片，直接内存处理
    )

    # 遍历每个检测结果
    for result in results:
        # 绘制检测框
        result_img = result.plot()  # BGR格式
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # 显示检测结果
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img_rgb)
        plt.axis('off')
        plt.title('YOLOv8 Detection Results')
        plt.show()

        # 打印检测到的目标
        print("\n检测到的对象:")
        if result.boxes is None or len(result.boxes) == 0:
            print("- 没有检测到任何目标")
        else:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"- 类别: {class_name} | 置信度: {confidence:.2f} | 坐标: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


if __name__ == "__main__":
    # 示例
    image_path = "shiyanqu4_9_12.jpg"  # 替换成你的图片路径
    model_path = "runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt"  # 替换成你的模型路径

    predict_single_image(
        image_path=image_path,
        model_path=model_path,
        conf_threshold=0.25,   # 置信度阈值
        iou_threshold=0.6     # NMS的IOU阈值
    )
