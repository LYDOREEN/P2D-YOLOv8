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
    def __init__(self, model_path, overlap_medium=0.5, tile_size_medium=512):
        self.model = YOLO(model_path)
        self.overlap_medium = overlap_medium  # 中图裁剪重叠度 (10240→512)
        self.tile_size_medium = tile_size_medium  # 512
        self.stride_medium = int(tile_size_medium * (1 - overlap_medium))  # 512*0.5=256

        # 类别颜色
        self.colors = {0: [255, 255, 0], 1: [0, 255, 0], 2: [0,255,255], 3: [128,0,128]} # 3: [255, 192, 203]
        # 新增的类别名映射
        self.class_names = {
            0: "tower",
            1: "building",
            2: "pole",
            3: "chimney"
        }

    def read_image_with_gdal(self, image_path):
        """使用 GDAL 读取图像"""
        dataset = gdal.Open(image_path)
        if dataset is None:
            raise ValueError("无法使用 GDAL 打开图像文件")

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

    def crop_medium_image(self, medium_image_path, output_dir):
        """裁剪 10240×10240 → 512×512 (重叠度 0.5)"""
        os.makedirs(output_dir, exist_ok=True)
        img_array = cv2.imread(medium_image_path)
        height, width = img_array.shape[:2]

        crops = []
        cols = math.ceil((width - self.tile_size_medium) / self.stride_medium) + 1
        rows = math.ceil((height - self.tile_size_medium) / self.stride_medium) + 1

        for i in tqdm(range(rows), desc="裁剪进度"):
            for j in range(cols):
                x1 = j * self.stride_medium
                y1 = i * self.stride_medium
                x2 = min(x1 + self.tile_size_medium, width)
                y2 = min(y1 + self.tile_size_medium, height)

                if x2 == width: x1 = width - self.tile_size_medium
                if y2 == height: y1 = height - self.tile_size_medium

                crop = img_array[y1:y2, x1:x2]
                crop_name = f"crop_{i}_{j}.tif"
                crop_path = os.path.join(output_dir, crop_name)
                cv2.imwrite(crop_path, crop)

                crops.append({
                    "path": crop_path,
                    "coords": (x1, y1, x2, y2),
                    "grid_pos": (i, j)
                })

        return crops

    def detect_tiles(self, crops, labels_dir, edge_margin=5, class_conf_thresholds=None):
        """检测 512×512 小图，并过滤边缘框 + 按类别置信度过滤"""
        if class_conf_thresholds is None:
            class_conf_thresholds = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # 默认阈值

        os.makedirs(labels_dir, exist_ok=True)
        detections = []

        for crop in tqdm(crops, desc="检测进度"):
            results = self.model(crop["path"])
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()

            # 保存 YOLO 格式标签
            label_path = os.path.join(labels_dir, f"{os.path.splitext(os.path.basename(crop['path']))[0]}.txt")
            with open(label_path, 'w') as f:
                for box_, conf, cls_id in zip(boxes, confs, cls_ids):
                    cls_id = int(cls_id)
                    if conf < class_conf_thresholds.get(cls_id, 0.3):
                        continue

                    # YOLO 格式: class_id x_center y_center width height (归一化)
                    img_h, img_w = 512, 512
                    x_center = ((box_[0] + box_[2]) / 2) / img_w
                    y_center = ((box_[1] + box_[3]) / 2) / img_h
                    width = (box_[2] - box_[0]) / img_w
                    height = (box_[3] - box_[1]) / img_h
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            x1_medium, y1_medium, x2_medium, y2_medium = crop["coords"]
            original_boxes = []

            for box_, conf, cls_id in zip(boxes, confs, cls_ids):
                cls_id = int(cls_id)
                # (1) 检查置信度是否达标
                if conf < class_conf_thresholds.get(cls_id, 0.3):  # 默认 0.3
                    continue

                # (2) 检查是否靠近边缘
                x1, y1, x2, y2 = box_
                if (x1 < edge_margin or y1 < edge_margin or
                        x2 > (512 - edge_margin) or y2 > (512 - edge_margin)):
                    continue  # 跳过边缘框

                # 转换到 10240×10240 坐标系
                global_x1 = x1 + x1_medium
                global_y1 = y1 + y1_medium
                global_x2 = x2 + x1_medium
                global_y2 = y2 + y1_medium

                original_boxes.append([global_x1, global_y1, global_x2, global_y2, conf, cls_id])

            detections.append({
                "grid_pos": crop["grid_pos"],
                "coords": crop["coords"],
                "boxes": original_boxes
            })

        return detections

    def apply_nms(self, boxes, class_conf_thresholds=None, class_nms_thresholds=None):
        """NMS 后处理（支持按类别设置不同阈值）"""
        if class_conf_thresholds is None:
            class_conf_thresholds = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # 默认置信度阈值
        if class_nms_thresholds is None:
            class_nms_thresholds = {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6}  # 默认 NMS 阈值

        if len(boxes) == 0:
            return []

        # 按类别分组
        class_groups = {}
        for box in boxes:
            cls_id = box[5]
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(box)

        nms_boxes = []
        for cls_id, cls_boxes in class_groups.items():
            boxes_xywh = []
            confidences = []

            for box in cls_boxes:
                x1, y1, x2, y2, conf, _ = box
                boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(conf)

            boxes_xywh = np.array(boxes_xywh).astype(np.float32)
            confidences = np.array(confidences).astype(np.float32)

            # 获取该类别的阈值
            conf_thresh = class_conf_thresholds.get(cls_id, 0.3)
            nms_thresh = class_nms_thresholds.get(cls_id, 0.3)

            indices = cv2.dnn.NMSBoxes(
                boxes_xywh.tolist(),
                confidences.tolist(),
                score_threshold=conf_thresh,
                nms_threshold=nms_thresh
            )

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes_xywh[i]
                    nms_boxes.append([x, y, x + w, y + h, confidences[i], cls_id])

        return nms_boxes

    def visualize_detections(self, image, boxes, output_path):
        """可视化检测框并保存"""
        alpha = 1
        img = image.copy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = map(float, box)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(cls_id)
            color = self.colors.get(cls_id, [0, 255, 0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            #label = f"{self.model.names[cls_id]} {conf:.2f}"
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            label = f"{class_name} {conf:.2f}"
            # 计算文本背景大小
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # 创建文本背景的覆盖层
            overlay = img.copy()
            cv2.rectangle(overlay,
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color, -1)

            # 将透明背景与原始图像混合
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            # 绘制文本背景
            #cv2.rectangle(img, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), color, -1)
            cv2.putText(img, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        #cv2.imwrite(output_path, img)
        save_with_georeference(output_path,img,r"D:\pytorch_learning\dataprocessing\shiyanqu1.tif")

    def process_10240_images(self, input_dir, output_dir):
        """直接处理10240×10240图像的完整流程"""
        # 定义不同类别的置信度阈值
        class_conf_thresholds = {
            0: 0.25,  # 类别0的置信度阈值
            1: 0.25,  # 类别1的置信度阈值
            2: 0.25,  # 类别2的置信度阈值
            3: 0.25  # 类别3的置信度阈值
        }

        # 定义不同类别的 NMS 阈值
        class_nms_thresholds = {
            0: 0.6,  # 类别0的NMS阈值
            1: 0.6,  # 类别1的NMS阈值
            2: 0.6,  # 类别2的NMS阈值
            3: 0.6  # 类别3的NMS阈值
        }

        # 获取所有10240×10240图像
        image_paths = glob(os.path.join(input_dir, "*.tif")) + glob(os.path.join(input_dir, "*.png")) + glob(os.path.join(input_dir, "*.jpg"))

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        medium_results_dir = os.path.join(output_dir, "medium_results")
        os.makedirs(medium_results_dir, exist_ok=True)

        all_final_boxes = []

        for img_path in tqdm(image_paths, desc="处理10240×10240图像"):
            img_name = os.path.basename(img_path)

            # 1. 裁剪 10240×10240 → 512×512
            medium_crops_dir = os.path.join(output_dir, f"medium_crops_{os.path.splitext(img_name)[0]}")
            medium_crops = self.crop_medium_image(img_path, medium_crops_dir)

            # 2. 检测 512×512 并保存标签
            labels_dir = os.path.join(medium_crops_dir, "labels")
            detections = self.detect_tiles(medium_crops, labels_dir, class_conf_thresholds=class_conf_thresholds)

            # 3. 合并回 10240×10240 + NMS
            merged_boxes = self.apply_nms(
                [box for det in detections for box in det["boxes"]],
                class_conf_thresholds=class_conf_thresholds,
                class_nms_thresholds=class_nms_thresholds
            )

            # 添加到全局结果
            all_final_boxes.extend(merged_boxes)

            # 4. 可视化 10240×10240 结果
            medium_img = cv2.imread(img_path)
            medium_output_path = os.path.join(medium_results_dir, img_name)
            self.visualize_detections(medium_img, merged_boxes, medium_output_path)

            # 5. 保存当前图像的检测结果
            with open(os.path.join(output_dir, f"detections_{os.path.splitext(img_name)[0]}.txt"), 'w') as f:
                for box in merged_boxes:
                    f.write(f"{box[5]} {box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

        print(f"处理完成！结果保存在: {output_dir}")

    def process_large_image(self, large_image_path, output_dir):
        """处理51200×51200单张大图完整流程"""
        class_conf_thresholds = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        class_nms_thresholds = {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6}

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(large_image_path))[0]
        crops_dir = os.path.join(output_dir, f"crops_{base_name}")
        labels_dir = os.path.join(crops_dir, "labels")
        vis_output_path = os.path.join(output_dir, f"visual_{base_name}.jpg")
        result_txt_path = os.path.join(output_dir, f"detections_{base_name}.txt")

        # 1. 读取并裁剪
        crops = self.crop_medium_image(large_image_path, crops_dir)

        # 2. 子图检测
        detections = self.detect_tiles(crops, labels_dir, class_conf_thresholds=class_conf_thresholds)

        # 3. 合并 + NMS
        merged_boxes = self.apply_nms(
            [box for det in detections for box in det["boxes"]],
            class_conf_thresholds=class_conf_thresholds,
            class_nms_thresholds=class_nms_thresholds
        )

        # 4. 可视化
        large_image = cv2.imread(large_image_path)
        self.visualize_detections(large_image, merged_boxes, vis_output_path)

        # 5. 保存检测框坐标
        with open(result_txt_path, 'w') as f:
            for box in merged_boxes:
                f.write(f"{box[5]} {box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

        print(f"完成！输出目录: {output_dir}")

def save_with_georeference(output_path, image_array, reference_path):
    """
    保存带地理参考的图像（如GeoTIFF），使用原始影像作为参考
    :param output_path: 输出路径
    :param image_array: 可视化后的图像（H x W x 3）
    :param reference_path: 原始影像路径（具有地理参考）
    """
    # 打开参考影像
    ref_ds = gdal.Open(reference_path)
    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()

    height, width, channels = image_array.shape

    # 创建输出影像
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, width, height, channels, gdal.GDT_Byte)

    # 设置地理参考信息
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)

    # 写入图像数据（按波段）
    for i in range(channels):
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(image_array[:, :, i])

    out_ds.FlushCache()
    out_ds = None

if __name__ == "__main__":
    processor = GDALRemoteImageProcessor(
        model_path="runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt",  # 替换为你的模型路径
        overlap_medium=0.5  # 10240→512 重叠度
    )

    # 直接处理10240×10240图像
    processor.process_10240_images(
        input_dir="shiyanqu1_10240",  # 10240×10240图像所在文件夹
        output_dir="output1"  # 输出目录
    )