from PIL import Image
import os

def crop_image_to_size(image_path, output_path, target_size):
    with Image.open(image_path) as img:
        width, height = img.size

        # 计算裁剪区域
        if width == 481 and height == 321:
            # 裁剪为 480x320
            left = 0.5
            top = 0
            right = width - 0.5
            bottom = height - 0.5
        elif width == 321 and height == 481:
            # 裁剪为 320x480
            left = 0
            top = 0.5
            right = width - 0.5
            bottom = height - 0.5
        else:
            # 如果分辨率不匹配，则不处理
            print(f"Image {image_path} does not match target sizes. Skipping.")
            return

        # 执行裁剪
        cropped_img = img.crop((left, top, right, bottom))

        # 保存裁剪后的图片
        cropped_img.save(output_path)
        print(f"Processed and saved: {output_path}")

def process_images(directory):
    for filename in os.listdir(directory):
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename)
            # 调用裁剪函数
            crop_image_to_size(image_path, output_path, (480, 320))

# 调用函数处理指定目录下的图片
directory = '/mnt/h/data/CBSD68-dataset/CBSD68/original_320'
process_images(directory)
