import os
import shutil

# ================== 配置区 ==================
dataset_root = r"D:/pytorch_learning/ultralytics-main/datasets/obstacle_7.0_yolo/test_scale_split/medium"
backup_root = r"D:/pytorch_learning/ultralytics-main/datasets/obstacle_7.0_yolo/test_scale_split/backup_class2"
remove_class_id = 2
img_exts = (".jpg", ".png", ".tif")
# ===========================================


def contains_target_class(label_path, class_id):
    with open(label_path, "r") as f:
        for line in f:
            if int(line.split()[0]) == class_id:
                return True
    return False


def main():
    img_dir = os.path.join(dataset_root, "images")
    lbl_dir = os.path.join(dataset_root, "labels")

    os.makedirs(os.path.join(backup_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(backup_root, "labels"), exist_ok=True)

    moved = 0

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith(img_exts):
            continue

        lbl_name = img_name.rsplit(".", 1)[0] + ".txt"
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if not os.path.exists(lbl_path):
            continue

        if contains_target_class(lbl_path, remove_class_id):
            shutil.move(img_path, os.path.join(backup_root, "images", img_name))
            shutil.move(lbl_path, os.path.join(backup_root, "labels", lbl_name))
            moved += 1

    print(f"完成：已移动包含 class={remove_class_id} 的样本 {moved} 张")


if __name__ == "__main__":
    main()
