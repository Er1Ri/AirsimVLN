from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("GroundingDINO-34ef00dcdf5cadb84c21b59db4fc42a4d4c75047/groundingdino/config/GroundingDINO_SwinT_OGC.py", "pretrained/groundingdino_swint_ogc.pth")
IMAGE_PATH = "color_image.jpg"
TEXT_PROMPT = "duck . coke . mirror ."
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