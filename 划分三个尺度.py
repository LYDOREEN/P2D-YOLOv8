import os
import shutil
from PIL import Image

# ================== 配置区 ==================
test_root = r"D:/pytorch_learning/ultralytics-main/datasets/obstacle_7.0_yolo/test"
out_root = r"D:/pytorch_learning/ultralytics-main/datasets/obstacle_7.0_yolo/test_scale_split"

# COCO尺度定义（像素面积）
SMALL_MAX = 32 * 32
MEDIUM_MAX = 96 * 96
# ===========================================


def get_box_area(label_line, img_w, img_h):
    """
    YOLO格式：cls cx cy w h（归一化）
    返回：bbox 像素面积
    """
    _, cx, cy, w, h = map(float, label_line.strip().split())
    box_w = w * img_w
    box_h = h * img_h
    return box_w * box_h


def judge_scale(areas):
    """
    判断一张图片属于哪个尺度
    规则：所有目标必须落在同一尺度
    """
    if all(a < SMALL_MAX for a in areas):
        return "small"
    elif all(SMALL_MAX <= a < MEDIUM_MAX for a in areas):
        return "medium"
    elif all(a >= MEDIUM_MAX for a in areas):
        return "large"
    else:
        return None  # 混合尺度，丢弃


def main():
    img_dir = os.path.join(test_root, "images")
    lbl_dir = os.path.join(test_root, "labels")

    # 创建输出目录
    for scale in ["small", "medium", "large"]:
        os.makedirs(os.path.join(out_root, scale, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_root, scale, "labels"), exist_ok=True)

    img_list = os.listdir(img_dir)

    stats = {"small": 0, "medium": 0, "large": 0, "discard": 0}

    for img_name in img_list:
        if not img_name.lower().endswith((".jpg", ".png", ".tif")):
            continue

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.rsplit(".", 1)[0] + ".txt")

        if not os.path.exists(lbl_path):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            continue

        areas = [get_box_area(line, img_w, img_h) for line in lines]
        scale = judge_scale(areas)

        if scale is None:
            stats["discard"] += 1
            continue

        shutil.copy(img_path, os.path.join(out_root, scale, "images", img_name))
        shutil.copy(lbl_path, os.path.join(out_root, scale, "labels", os.path.basename(lbl_path)))
        stats[scale] += 1

    print("尺度划分完成：")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
