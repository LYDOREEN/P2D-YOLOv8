import os
import glob
import cv2
import numpy as np
from collections import defaultdict

# ===================== 修改为你的数据集根目录 =====================
DATASET_ROOT = r"datasets/obstacle_7.0_yolo"
# ================================================================

subsets = ["train", "val", "test", "testA", "testB"]
class_names = ["tower", "building", "pole", "chimney"]

area_dict = defaultdict(list)
aspect_dict = defaultdict(list)
size_count = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
total_count = defaultdict(int)

total_images = 0
total_objects = 0

# 支持的图片格式
img_extensions = ["*.jpg", "*.png", "*.tif", "*.tiff"]

for subset in subsets:
    image_dir = os.path.join(DATASET_ROOT, subset, "images")
    label_dir = os.path.join(DATASET_ROOT, subset, "labels")

    if not os.path.exists(image_dir):
        continue

    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    total_images += len(image_paths)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"读取失败: {img_path}")
            continue

        H, W = img.shape[:2]

        label_path = os.path.join(
            label_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split()
            if len(data) != 5:
                continue

            cls, xc, yc, bw, bh = data
            cls = int(cls)
            bw, bh = float(bw), float(bh)

            box_w = bw * W
            box_h = bh * H
            area = box_w * box_h

            if min(box_w, box_h) == 0:
                continue

            # 推荐使用长边/短边
            aspect_ratio = max(box_w, box_h) / min(box_w, box_h)

            area_dict[cls].append(area)
            aspect_dict[cls].append(aspect_ratio)
            total_count[cls] += 1
            total_objects += 1

            # COCO 尺度划分
            if area < 32**2:
                size_count[cls]["small"] += 1
            elif area < 96**2:
                size_count[cls]["medium"] += 1
            else:
                size_count[cls]["large"] += 1


# ===================== 输出统计 =====================

print("\n========== 数据集总体信息 ==========")
print(f"总图像数: {total_images}")
print(f"总目标数: {total_objects}")
print()

print("| Class    | Avg Area (px²) | Avg Aspect Ratio | Small (%) | Medium (%) | Large (%) |")
print("|----------|----------------|------------------|-----------|------------|-----------|")

for cls_id, cls_name in enumerate(class_names):
    if total_count[cls_id] == 0:
        print(f"| {cls_name:<8} | 0 | 0 | 0 | 0 | 0 |")
        continue

    avg_area = np.mean(area_dict[cls_id])
    avg_aspect = np.mean(aspect_dict[cls_id])

    small_ratio = size_count[cls_id]["small"] / total_count[cls_id] * 100
    medium_ratio = size_count[cls_id]["medium"] / total_count[cls_id] * 100
    large_ratio = size_count[cls_id]["large"] / total_count[cls_id] * 100

    print(f"| {cls_name:<8} | {avg_area:.2f} | {avg_aspect:.2f} | "
          f"{small_ratio:.2f} | {medium_ratio:.2f} | {large_ratio:.2f} |")