import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch
import torchvision.ops as ops
from osgeo import gdal


class LargeImageDetector:
    def __init__(self, model_path, output_dir="output", tile_size=512, stride=512,border_thres=5, iou_thres=0.5, class_names=None):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.tile_size = tile_size
        self.stride = stride
        self.border_thres = border_thres
        self.iou_thres = iou_thres
        self.class_names = class_names
        self.colors = [
            (255, 255, 0),
            (0, 255, 0),
            (255, 13, 9),
            (255, 192, 203)
        ]

    def read_image_with_gdal(self, image_path):
        dataset = gdal.Open(image_path)
        if dataset is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount

        img = np.zeros((height, width, bands), dtype=np.uint8)
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            img[:, :, i] = band.ReadAsArray()
        transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        return img, transform, projection

    def apply_nms(self, boxes, scores, classes):
        if len(boxes) == 0:
            return [], [], []

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores)
        keep = ops.nms(boxes_tensor, scores_tensor, self.iou_thres)

        final_boxes = [boxes[i] for i in keep]
        final_scores = [scores[i] for i in keep]
        final_classes = [classes[i] for i in keep]

        return final_boxes, final_scores, final_classes

    def visualize_detections(self, image, boxes, scores, classes):
        img_vis = image.copy()
        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_id}:{score:.2f}"
            if self.class_names and cls_id < len(self.class_names):
                label = f"{self.class_names[cls_id]}:{score:.2f}"
            cv2.putText(img_vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img_vis

    def process_large_image(self, image_path):
        img_array, geo_transform, _ = self.read_image_with_gdal(image_path)
        height, width = img_array.shape[:2]
        print(height)
        print(width)

        all_boxes = []
        all_scores = []
        all_classes = []

        print(f"Processing large image: {image_path} ({width}x{height})")

        for y1 in tqdm(range(0, height, self.stride), desc="Sliding Y"):
            for x1 in range(0, width, self.stride):
                x2 = min(x1 + self.tile_size, width)
                y2 = min(y1 + self.tile_size, height)

                x1 = max(0, x2 - self.tile_size)
                y1 = max(0, y2 - self.tile_size)

                tile = img_array[y1:y2, x1:x2]
                if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                    continue

                results = self.model(tile, verbose=False)[0]
                boxes = results.boxes
                if boxes is None or boxes.xyxy.shape[0] == 0:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    score = box.conf.item()
                    cls_id = int(box.cls.item())

                    x_min, y_min, x_max, y_max = xyxy
                    if (x_min < self.border_thres or y_min < self.border_thres or
                        x_max > self.tile_size - self.border_thres or
                        y_max > self.tile_size - self.border_thres):
                        continue

                    global_box = [x1 + x_min, y1 + y_min, x1 + x_max, y1 + y_max]
                    all_boxes.append(global_box)
                    all_scores.append(score)
                    all_classes.append(cls_id)

        boxes_nms, scores_nms, classes_nms = self.apply_nms(all_boxes, all_scores, all_classes)

        output_dir = os.path.join(self.output_dir, "large_result")
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_with_detections = self.visualize_detections(img_array, boxes_nms, scores_nms, classes_nms)

        cv2.imwrite(os.path.join(output_dir, f"{image_name}_detect.jpg"), image_with_detections)

        label_path = os.path.join(output_dir, f"{image_name}.txt")
        with open(label_path, 'w') as f:
            for box, cls_id in zip(boxes_nms, classes_nms):
                x1, y1, x2, y2 = box
                xc = (x1 + x2) / 2 / width
                yc = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        print(f"[Done] Large image detection complete. Saved to: {output_dir}")

# 类别名列表（可选）
class_names = ['tower', 'building', 'pole', 'chimney']

detector = LargeImageDetector(
    model_path='runs/7.0/yolov8-p2-C2f_ODConv_7.0/weights/best.pt',
    output_dir='output',
    tile_size=512,
    stride=256,
    border_thres=5,
    iou_thres=0.5,
    class_names=class_names  # 用于可视化时显示类别名
)

detector.process_large_image(r'D:\pytorch_learning\dataprocessing\shiyanqu1.tif')
