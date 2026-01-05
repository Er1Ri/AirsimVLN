import torch
from PIL import Image
import matplotlib.pyplot as plt

from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    image_path = "recognize-anything/images/demo/demo1.jpg"
    im = Image.open(image_path).convert("RGB")

    # 可视化（可选）
    plt.figure()
    plt.imshow(im)
    plt.axis("off")
    plt.show()

    # 关键：get_transform 返回的是 transform 对象
    transform = get_transform(image_size=384)

    model = ram(
        pretrained="pretrained/ram_swin_large_14m.pth",
        image_size=384,
        vit="swin_l",
    ).to(device)
    model.eval()

    with torch.no_grad():
        image_tensor = transform(im).unsqueeze(0).to(device)
        tags_en, tags_zh = inference(image_tensor, model)

    print("Image Tags:", tags_en)
    print("图像标签:", tags_zh)

if __name__ == "__main__":
    main()

