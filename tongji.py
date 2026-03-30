import os
from glob import glob
from collections import defaultdict

def count_from_labels(label_dir, names_dict=None):
    label_files = glob(os.path.join(label_dir, '*.txt'))

    class_to_images = defaultdict(set)
    class_to_count = defaultdict(int)

    for label_path in label_files:
        image_id = os.path.splitext(os.path.basename(label_path))[0]
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() == '':
                    continue
                class_id = int(line.strip().split()[0])
                class_to_images[class_id].add(image_id)
                class_to_count[class_id] += 1

    # 打印统计结果
    print(f"{'类别':<20} {'图片数':<10} {'目标数':<10}")
    print("-" * 40)
    all_classes = sorted(set(list(class_to_images.keys()) + list(class_to_count.keys())))
    for class_id in all_classes:
        class_name = names_dict[class_id] if names_dict else str(class_id)
        image_count = len(class_to_images[class_id])
        object_count = class_to_count[class_id]
        print(f"{class_name:<20} {image_count:<10} {object_count:<10}")


# 🛠 类别名称（根据你的实际顺序填写）
names = {
    0: 'electric_tower',
    1: 'building',
    2: 'telegraph_pole',
    3: 'chimney',
}

# 🛠 标签路径
label_path = 'datasets/obstacle_7.0_yolo/test4_v3/labels'

count_from_labels(label_path, names_dict=names)
