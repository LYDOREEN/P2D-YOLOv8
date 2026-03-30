import os
import argparse
from tqdm import tqdm


def check_test_set(test_dir):
    """
    专门检查测试集的图片和标签对应关系

    参数:
        test_dir (str): 测试集目录路径（应包含images和labels子目录）
    """
    # 定义路径
    images_dir = os.path.join(test_dir, 'images')
    labels_dir = os.path.join(test_dir, 'labels')

    # 验证目录结构
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"测试集图片目录不存在: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"测试集标签目录不存在: {labels_dir}")

    print(f"\n🔍 正在检查测试集: {test_dir}")

    # 获取文件名（不带扩展名）
    image_bases = {os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    label_bases = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)
                   if f.endswith('.txt')}

    # 检查对应关系
    only_images = image_bases - label_bases
    only_labels = label_bases - image_bases
    matched = image_bases & label_bases

    print("\n📊 检查结果:")
    print(f"测试图片数量: {len(image_bases)}")
    print(f"测试标签数量: {len(label_bases)}")
    print(f"匹配成功的文件对: {len(matched)}")

    # 打印不匹配项（最多显示5个）
    if only_images:
        print(f"\n❌ 缺少标签的图片 ({len(only_images)}个):")
        for name in list(only_images)[:5]:
            print(f"  - {name}.jpg")
        if len(only_images) > 5:
            print(f"  ...(仅显示前5个，共{len(only_images)}个)")

    if only_labels:
        print(f"\n❌ 缺少图片的标签 ({len(only_labels)}个):")
        for name in list(only_labels)[:5]:
            print(f"  - {name}.txt")
        if len(only_labels) > 5:
            print(f"  ...(仅显示前5个，共{len(only_labels)}个)")

    # 快速内容检查
    if matched:
        print("\n⏳ 快速扫描标签内容...")
        empty_labels = []
        for base in tqdm(list(matched)[:1000], desc="抽样检查"):  # 最多检查1000个
            label_path = os.path.join(labels_dir, base + '.txt')
            if os.path.getsize(label_path) == 0:
                empty_labels.append(base)

        if empty_labels:
            print(f"⚠️ 发现{len(empty_labels)}个空标签文件 (示例):")
            for base in empty_labels[:3]:
                print(f"  - {base}.txt")

    # 最终结论
    if not only_images and not only_labels:
        print("\n✅ 测试集完整！所有图片和标签匹配")
    else:
        print("\n❌ 测试集存在问题，请修复不匹配项")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO测试集完整性检查')
    parser.add_argument('test_dir', type=str, help='测试集目录路径（包含images和labels子目录）')
    args = parser.parse_args()

    check_test_set(args.test_dir)