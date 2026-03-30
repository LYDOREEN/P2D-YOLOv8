import os
import cv2
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from ultralytics import YOLO
import math
import random
import warnings
from glob import glob

warnings.filterwarnings("ignore", category=UserWarning)


class GDALRemoteImageProcessor:
    def __init__(self, model_path, overlap=0.5, tile_size=512):
        self.model = YOLO(model_path)
        self.overlap = overlap
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap))

        # 为每个类别生成随机颜色
        self.colors = {0: [255, 255, 0], 1: [0, 255, 0], 2: [255, 13, 9], 3: [255, 192, 203]}

    def read_image_with_gdal(self, image_path):
        """使用GDAL读取图像"""
        dataset = gdal.Open(image_path)
        if dataset is None:
            raise ValueError("无法使用GDAL打开图像文件")

        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount

        if bands >= 3:
            band1 = dataset.GetRasterBand(1).ReadAsArray()
            band2 = dataset.GetRasterBand(2).ReadAsArray()
            band3 = dataset.GetRasterBand(3).ReadAsArray()
            img_array = np.dstack((band1, band2, band3))
        else:
            img_array = dataset.GetRasterBand(1).ReadAsArray()
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)

        return img_array, width, height

    def crop_image(self, image_path, output_dir):
        """使用GDAL将大图像裁剪为重叠的小瓦片"""
        os.makedirs(output_dir, exist_ok=True)
        img_array, width, height = self.read_image_with_gdal(image_path)
        print(f"原始图像尺寸: {width}x{height}")

        crops = []
        cols = math.ceil((width - self.tile_size) / self.stride) + 1
        rows = math.ceil((height - self.tile_size) / self.stride) + 1

        for i in tqdm(range(rows), desc="裁剪进度"):
            for j in range(cols):
                x1 = j * self.stride
                y1 = i * self.stride
                x2 = min(x1 + self.tile_size, width)
                y2 = min(y1 + self.tile_size, height)

                if x2 == width: x1 = width - self.tile_size
                if y2 == height: y1 = height - self.tile_size

                crop = img_array[y1:y2, x1:x2]
                crop_name = f"crop_{i}_{j}.tif"
                crop_path = os.path.join(output_dir, crop_name)
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                crops.append({
                    "path": crop_path,
                    "coords": (x1, y1, x2, y2),
                    "grid_pos": (i, j)
                })

        return crops

    def detect_tiles(self, crops, detection_dir, labels_dir):
        """对裁剪后的瓦片进行目标检测"""
        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)  # 创建标签文件夹
        detections = []

        for crop in tqdm(crops, desc="检测进度"):
            results = self.model(crop["path"])
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()

            x1, y1, x2, y2 = crop["coords"]
            original_boxes = []

            # 保存YOLO格式标签
            label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(crop["path"]))[0] + '.txt')
            with open(label_path, 'w') as f:
                for box_, conf, cls_id in zip(boxes, confs, cls_ids):
                    # YOLO格式: class_id x_center y_center width height (归一化)
                    img_h, img_w = cv2.imread(crop["path"]).shape[:2]
                    x_center = ((box_[0] + box_[2]) / 2) / img_w
                    y_center = ((box_[1] + box_[3]) / 2) / img_h
                    width = (box_[2] - box_[0]) / img_w
                    height = (box_[3] - box_[1]) / img_h
                    f.write(f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    # 保存原始坐标的框
                    original_boxes.append([
                        box_[0] + x1,
                        box_[1] + y1,
                        box_[2] + x1,
                        box_[3] + y1,
                        conf,
                        int(cls_id)
                    ])

            detections.append({
                "grid_pos": crop["grid_pos"],
                "coords": crop["coords"],
                "boxes": original_boxes
            })

            # 保存带标准YOLO标签的检测图像
            # detected_img = self.draw_yolo_boxes(cv2.imread(crop["path"]), boxes, confs, cls_ids)
            # detected_path = os.path.join(detection_dir, os.path.basename(crop["path"]))
            # cv2.imwrite(detected_path, detected_img)

        return detections

    def draw_yolo_boxes(self, image, boxes, confs, cls_ids):
        """绘制YOLO风格的标准标签框"""
        img = image.copy()
        for box_, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box_)
            cls_id = int(cls_id)

            # 获取类别颜色
            color = self.colors.get(cls_id, (0, 255, 0))

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 创建标签文本
            label = f"{self.model.names[cls_id]} {conf:.2f}"

            # 计算文本背景大小
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制文本背景
            cv2.rectangle(img, (x1, y1 - text_height - 10),
                          (x1 + text_width, y1), color, -1)

            # 绘制文本
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def merge_detections(self, detections, original_size, edge_width, decay_factor,class_conf_thresholds):
        """
        合并重叠区域的检测框（带动态边缘衰减）

        参数:
            detections: 检测结果列表
            original_size: 原始图像尺寸 (width, height)
            edge_width: 边缘区域宽度（像素）
            decay_factor: 基础衰减因子（0-1）
        """
        all_boxes = []
        img_width, img_height = original_size

        # 设置默认置信度阈值
        if class_conf_thresholds is None:
            class_conf_thresholds = {}
        default_conf_thresh = 0.1

        for detection in detections:
            for box in detection["boxes"]:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)  # 确保类别ID是整数
                # 计算到各边缘的最小距离
                edge_dist = min(x1, y1, img_width - x2, img_height - y2)

                # 动态边缘衰减
                if edge_dist < edge_width:
                    # 非线性衰减：距离边缘越近衰减越大
                    decay = decay_factor ** (1 + (edge_width - edge_dist) / edge_width)
                    conf *= decay

                all_boxes.append([x1, y1, x2, y2, conf, cls_id])

        # 按类别分组
        class_groups = {}
        for box in all_boxes:
            cls_id = box[5]
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(box)

        merged_boxes = []

        # 对每个类别的框进行NMS合并
        for cls_id, boxes in class_groups.items():
            boxes_array = np.array([b[:4] + [b[4]] for b in boxes])

            if len(boxes_array) > 0:
                boxes_xywh = []
                for box in boxes_array:
                    x1, y1, x2, y2, conf = box
                    boxes_xywh.append([x1, y1, x2 - x1, y2 - y1, conf])

                boxes_xywh = np.array(boxes_xywh)
                confidences = boxes_xywh[:, 4]
                boxes_xywh = boxes_xywh[:, :4].astype(np.float32)
                # 获取该类别的置信度阈值
                cls_conf_thresh = class_conf_thresholds.get(cls_id, default_conf_thresh)
                indices = cv2.dnn.NMSBoxes(
                    boxes_xywh.tolist(),
                    confidences.tolist(),
                    score_threshold=cls_conf_thresh,
                    nms_threshold=0.3
                )

                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes_xywh[i]
                        merged_boxes.append([
                            x, y, x + w, y + h,
                            confidences[i],
                            cls_id
                        ])

        return merged_boxes

    '''
    def merge_detections(self, detections, original_size):
        """合并重叠区域的检测框"""
        all_boxes = []
        for detection in detections:
            all_boxes.extend(detection["boxes"])

        # 按类别分组
        class_groups = {}
        for box_ in all_boxes:
            cls_id = box_[5]
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(box_)

        merged_boxes = []

        # 对每个类别的框进行NMS合并
        for cls_id, boxes in class_groups.items():
            boxes_array = np.array([b[:4] + [b[4]] for b in boxes])

            if len(boxes_array) > 0:
                boxes_xywh = []
                for box_ in boxes_array:
                    x1, y1, x2, y2, conf = box_
                    boxes_xywh.append([x1, y1, x2 - x1, y2 - y1, conf])

                boxes_xywh = np.array(boxes_xywh)
                confidences = boxes_xywh[:, 4]
                boxes_xywh = boxes_xywh[:, :4].astype(np.float32)

                indices = cv2.dnn.NMSBoxes(
                    boxes_xywh.tolist(),
                    confidences.tolist(),
                    score_threshold=0.1,
                    nms_threshold=0.5
                )

                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes_xywh[i]
                        merged_boxes.append([
                            x, y, x + w, y + h,
                            confidences[i],
                            cls_id
                        ])

        return merged_boxes
    '''

    def visualize_edge_aware(self, image_path, merged_boxes, output_path, edge_width=15):
        """
               边缘感知可视化（显示边缘衰减效果）

               参数:
                   image_path: 原始图像路径
                   merged_boxes: 合并后的检测框
                   output_path: 输出图像路径
                   edge_width: 边缘区域宽度
               """
        try:
            img_array, width, height = self.read_image_with_gdal(image_path)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # 绘制边缘区域示意（使用整数坐标）
            cv2.rectangle(img, (0, 0), (width - 1, edge_width - 1), (0, 0, 200), 1)  # 上边缘
            cv2.rectangle(img, (0, 0), (edge_width - 1, height - 1), (0, 0, 200), 1)  # 左边缘
            cv2.rectangle(img, (width - edge_width, 0), (width - 1, height - 1), (0, 0, 200), 1)  # 右边缘
            cv2.rectangle(img, (0, height - edge_width), (width - 1, height - 1), (0, 0, 200), 1)  # 下边缘

            for box in merged_boxes:
                # 确保坐标是整数
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cls_id = int(box[5])
                color = self.colors.get(cls_id, (0, 255, 0))

                # 检查是否是边缘框
                edge_dist = min(x1, y1, width - x2, height - y2)
                is_edge = edge_dist < edge_width

                # 绘制边界框（边缘框用虚线）
                thickness = 2
                if is_edge:
                    # 虚线效果
                    for i in range(0, x2 - x1, 8):
                        cv2.line(img, (x1 + i, y1), (x1 + min(i + 4, x2 - x1), y1), color, thickness)
                        cv2.line(img, (x1 + i, y2 - 1), (x1 + min(i + 4, x2 - x1), y2 - 1), color, thickness)
                    for i in range(0, y2 - y1, 8):
                        cv2.line(img, (x1, y1 + i), (x1, y1 + min(i + 4, y2 - y1)), color, thickness)
                        cv2.line(img, (x2 - 1, y1 + i), (x2 - 1, y1 + min(i + 4, y2 - y1)), color, thickness)
                else:
                    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), color, thickness)

                # 显示边缘信息
                if is_edge:
                    text = f"{edge_dist}px"  # 显示到边缘的距离
                    cv2.putText(img, text, (x1, max(15, y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 添加图例
            legend = f"Edge Zone: {edge_width}px (Blue Border)"
            cv2.putText(img, legend, (10, height - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"边缘感知可视化失败: {str(e)}")
            raise

    def visualize_results(self, image_path, merged_boxes, output_path):
        """可视化最终结果，使用YOLO标准标签风格"""
        img_array, _, _ = self.read_image_with_gdal(image_path)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        for box_ in merged_boxes:
            x1, y1, x2, y2, conf, cls_id = box_
            color = self.colors.get(cls_id, (0, 255, 0))

            # 绘制边界框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 创建标签文本
            label = f"{self.model.names[cls_id]} {conf:.2f}"

            # 计算文本背景大小
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制文本背景
            cv2.rectangle(img, (int(x1), int(y1) - text_height - 10),(int(x1) + text_width, int(y1)), color, -1)

            # 绘制文本
            cv2.putText(img, label, (int(x1), int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, img)

    def visualize_results1(self, image_path, merged_boxes, output_path):
        """可视化最终结果，只绘制边界框"""
        img_array, _, _ = self.read_image_with_gdal(image_path)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        for box_ in merged_boxes:
            x1, y1, x2, y2, conf, cls_id = box_
            color = self.colors.get(cls_id, (0, 255, 0))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.imwrite(output_path, img)

    def process_image(self, image_path, output_base_dir, edge_width, decay_factor, class_conf_thresholds):
        """完整处理流程（带边缘处理）"""
        crop_dir = os.path.join(output_base_dir, "crops")
        detection_dir = os.path.join(output_base_dir, "detections")
        labels_dir = os.path.join(output_base_dir, "labels")  # 新增标签目录
        result_dir = os.path.join(output_base_dir, "results")

        print("开始裁剪图像...")
        crops = self.crop_image(image_path, crop_dir)

        print("开始检测瓦片...")
        detections = self.detect_tiles(crops, detection_dir, labels_dir)  # 传入labels_dir

        print("开始合并检测结果(带边缘衰减)...")
        img_array, width, height = self.read_image_with_gdal(image_path)
        merged_boxes = self.merge_detections(
            detections,
            (width, height),
            edge_width=edge_width,
            decay_factor=decay_factor,
            class_conf_thresholds=class_conf_thresholds
        )

        print("生成最终结果...")
        # 标准可视化
        standard_path = os.path.join(result_dir, "final_result_standard.tif")
        os.makedirs(result_dir, exist_ok=True)
        self.visualize_results(image_path, merged_boxes, standard_path)

        # 边缘感知可视化
        #edge_aware_path = os.path.join(result_dir, "final_result_edge_aware.tif")
        #self.visualize_edge_aware(image_path, merged_boxes, edge_aware_path, edge_width)

        print(f"处理完成! 结果保存在: {result_dir}")


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = GDALRemoteImageProcessor(
        model_path="runs/train/yolov8-p2-C2f_ODConv/weights/best.pt",  # 替换为你的模型路径
        overlap=0.5,
        tile_size=512
    )

    # 设置各类别的置信度阈值
    class_thresholds = {0: 0.5, 1: 0.6, 2: 0.3, 3: 0.5}

    # 输入文件夹路径
    input_folder = r"D:\pytorch_learning\dataprocessing\shiyanqu1_10240"

    # 获取文件夹中所有.tif文件
    image_files = glob(os.path.join(input_folder, "*.tif"))

    # 处理每张图片
    for image_path in image_files:
        # 从文件名生成输出目录名（如shiyanqu1_1_1.tif -> result1_1_1）
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # 获取不带扩展名的文件名
        output_dir = f"result{base_name.split('shiyanqu1_')[-1]}"  # 生成类似result1_1_1的目录名

        print(f"\n正在处理图像: {image_path}")
        print(f"输出目录: {output_dir}")

        # 处理当前图像
        processor.process_image(
            image_path=image_path,
            output_base_dir=output_dir,
            edge_width=1,
            decay_factor=0.5,
            class_conf_thresholds=class_thresholds
        )

    print("\n所有图像处理完成！")

