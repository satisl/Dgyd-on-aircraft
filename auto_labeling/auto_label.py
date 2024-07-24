import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.utils.plotting import Annotator, colors
from groundingdino.util.inference import Model
import supervision as sv
import torchvision
import torch


# yolo detect objects
def yolo_detect(image, model, conf, iou):
    results = model.predict(
        source=image,
        imgsz=640,
        half=True,
        device="cuda:0",
        save=False,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    r = results[0]
    xyxy = r.obb.xyxy
    clss = r.obb.cls.cpu().tolist()
    names = r.names
    return xyxy, clss, names


# grounding_dino detect
def grounding_dino(model, image, classes, box_threshold, text_threshold, nms_threshold):
    detections = model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = (
        torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold,
        )
        .numpy()
        .tolist()
    )

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")
    return detections


# Prompting SAM with detected boxes
def segment(sam_predictor: SAMPredictor, image: np.ndarray, xyxy: np.ndarray) -> list:
    sam_predictor.set_image(image)
    masks = []
    for box in xyxy:
        results = sam_predictor(bboxes=box)
        masks.append(results[0].masks.xy[0])
    sam_predictor.reset_image()
    return masks


def show(image, masks, clss, names):
    annotator = Annotator(image)
    for mask, cls in zip(masks, clss):
        annotator.seg_bbox(
            mask=mask, mask_color=colors(int(cls)), det_label=names[int(cls)]
        )
    cv2.imshow("0", cv2.resize(annotator.im, (show_width, show_height)))
    cv2.waitKey(1)


def save(image_name, masks, clss, names):
    # 根据掩码生成polygon，并写入json标注文件
    shapes = []
    with open(f'{labels_path}/{image_name.rsplit(".", 1)[0]}.json', mode="w") as f:
        for mask, cls in zip(masks, clss):
            points = []
            for point in mask.tolist():
                points.append(point)
            shape = {
                "label": f"{names[int(cls)]}",
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
            }
            shapes.append(shape)
        height, width = image.shape[:2]

        json_data = {
            "version": "2.3.4",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        json.dump(json_data, f, indent=2)


labels_path = (
    r"../datasets/segment/origin1/json"
)
images_path = (
    r"../datasets/segment/origin1/images"
)
os.makedirs(labels_path, exist_ok=False)
show_width = 640
show_height = 640

# segment anything
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=1024,
    model=r"./other/mobile_sam.pt",
    verbose=False,
    save=False,
)
predictor = SAMPredictor(overrides=overrides)
print("sam部署完毕")

# yolo
detect_path = r"../yolov8/obb_480dataset_imgsz640_v8n_SGD/weights/best.pt"
yolo_model = YOLO(detect_path, task="detect")
conf = 0.5
iou = 0.5
print("yolo部署完毕")

# # grounding dino
# GROUNDING_DINO_CONFIG_PATH = "other/GroundingDINO_SwinB_cfg.py"
# GROUNDING_DINO_CHECKPOINT_PATH = "other/groundingdino_swinb_cogcoor.pth"

# grounding_dino_model = Model(
#     model_config_path=GROUNDING_DINO_CONFIG_PATH,
#     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
# )

# BOX_THRESHOLD = 0.25
# TEXT_THRESHOLD = 0.25
# NMS_THRESHOLD = 0.1

# CLASSES = ["number"]
# print("grounding dino部署完毕")

frames_num = 0
pre_time = cv2.getTickCount()
# load image
for i in os.listdir(images_path):
    image = cv2.imread(f"{images_path}/{i}")
    if "yolo_model" in locals():
        xyxy, clss, names = yolo_detect(image, yolo_model, conf, iou)
    else:
        detections = grounding_dino(
            grounding_dino_model,
            image,
            CLASSES,
            BOX_THRESHOLD,
            TEXT_THRESHOLD,
            NMS_THRESHOLD,
        )
        xyxy = detections.xyxy
        clss = [cls if cls is not None else -1 for cls in detections.class_id]
        names = CLASSES + ["None"]

    masks = segment(
        sam_predictor=predictor, image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyxy=xyxy
    )
    show(image, masks, clss, names)
    save(i, masks, clss, names)
    frames_num += 1
    if frames_num > 60:
        print(
            f"fps:{frames_num * cv2.getTickFrequency() / (cv2.getTickCount() - pre_time):.2f}"
        )
        frames_num = 0
        pre_time = cv2.getTickCount()
