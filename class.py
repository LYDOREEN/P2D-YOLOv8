import os

label_path = "labels"  # 修改为你的标签路径

# 初始化统计字典
object_counts = {}  # 记录每个类别的目标总数
image_counts = {}  # 记录包含每个类别的图片数
unique_images_per_class = {}  # 记录每个类别对应的唯一图片名（用于去重）

for file in os.listdir(label_path):
    with open(os.path.join(label_path, file)) as f:
        # 获取当前图片中的类别集合（去重）
        classes_in_image = set()
        for line in f:
            cls = int(float(line.strip().split()[0]))
            classes_in_image.add(cls)
            object_counts[cls] = object_counts.get(cls, 0) + 1

        # 更新包含各类别的图片数
        for cls in classes_in_image:
            image_counts[cls] = image_counts.get(cls, 0) + 1
            if cls not in unique_images_per_class:
                unique_images_per_class[cls] = set()
            unique_images_per_class[cls].add(file)

# 打印统计结果（按类别ID排序）
print("Class\tObjects\tImages\tUnique Images")
print("-----------------------------------")
for cls_id in sorted(object_counts):
    print(
        f"{cls_id}\t{object_counts[cls_id]}\t{image_counts.get(cls_id, 0)}\t{len(unique_images_per_class.get(cls_id, set()))}")

# 可选：保存详细统计到CSV
import csv

with open('class_stats.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class', 'Objects', 'Images', 'Unique Images'])
    for cls_id in sorted(object_counts):
        writer.writerow([cls_id, object_counts[cls_id], image_counts.get(cls_id, 0),
                         len(unique_images_per_class.get(cls_id, set()))])
