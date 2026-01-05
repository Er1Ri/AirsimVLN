import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from PIL import Image
import matplotlib.pyplot as plt

from ram.models import ram
from ram import get_transform, inference_ram  # 这行很重要！

# 1. 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 读原图
image_path = "recognize-anything/images/demo/demo1.jpg"
im = Image.open(image_path).convert("RGB")
plt.imshow(im)
plt.axis("off")
plt.show()

# 3. 加载模型（注意权重路径要存在）
model = ram(
    pretrained="pretrained/ram_swin_large_14m.pth",  # 这里改成你实际下载的位置
    image_size=384,
    vit="swin_l"
)
model = model.to(device)
model.eval()

# 4. 构造 transform，并对图像做预处理
transform = get_transform(image_size=384)
image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

# 5. 推理
res = inference_ram(image_tensor, model)

print("Image Tags:", res[0])         # 英文标签字符串
print("Tag List:", res[0].split(" | "))  # 拆成列表



from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

#注意确定文件位置
model = load_model("GroundingDINO-34ef00dcdf5cadb84c21b59db4fc42a4d4c75047/groundingdino/config/GroundingDINO_SwinT_OGC.py", "pretrained/groundingdino_swint_ogc.pth")
IMAGE_PATH = "recognize-anything/images/demo/demo1.jpg"
TEXT_PROMPT = res[0]
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)

pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

plt.imshow(pil_image)
plt.show()