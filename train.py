import gc
from ultralytics import YOLO
import os
from PIL import Image

project_name = 'yolov8'

class_ = 'obb'
dataset_name = fr'D:\Double-digit-yolo-detection-on-aircraft\datasets\{class_}\{class_}.yaml'
imgsz = 640
optimizer = 'SGD'
model_name = f'{class_}_400dataset_imgsz{imgsz}_v8n_{optimizer}'
batchsz = -1  # 96:500 160:300
task = 'obb'

if __name__ == '__main__':
    # # train and val
    model = YOLO('yolov8n-obb.yaml')
    model.train(data=dataset_name, epochs=50, verbose=True,
                imgsz=imgsz, batch=batchsz, cache='disk',
                pretrained=False, optimizer=optimizer,
                amp=True, val=True, save_period=50, patience=
                100, name=model_name, project=project_name,
                fliplr=0)

    model = YOLO(
        r'D:\Double-digit-yolo-detection-on-aircraft\yolov8\detect_400dataset_imgsz640_v8n_SGD/weights/last.pt')
    model.train(resume=True)

    del model
    gc.collect()


    # 用训练保存的各个epoch的模型val一遍test集

    ckpt = os.listdir(f'{project_name}/{model_name}/weights')

    ckpt.remove('best.pt')
    ckpt.remove('last.pt')
    ckpt.sort(key=lambda x: int(x[5:][:-3]))
    ckpt.extend(['last.pt', 'best.pt'])

    metrics = ['F1_curve.png', 'P_curve.png', 'PR_curve.png', 'R_curve.png']
    height = 1500
    width = 2250

    for id1, i in enumerate(ckpt):

        model = YOLO(f'{project_name}/{model_name}/weights/{i}')
        model.val(data=dataset_name, split='test', imgsz=imgsz, batch=1, half=True,
                  plots=True, iou=0.5, name=f'{model_name}--{i[:-3]}')
        del model
        gc.collect()

        image = Image.new("RGB", (len(metrics) * width, height), "white")

        # 合并各个epoch模型val出来的F1_curve,P_curve,PR_curve,R_curve图
        for id2, j in enumerate(metrics):
            image.paste(Image.open(f'{os.getcwd()}/runs/{task}/{model_name}--{i[:-3]}/{j}'), (id2 * width, 0))

        image.save(f'{project_name}/{model_name}/{i[:-3]}_test.png')

    image = Image.new("RGB", (len(metrics) * width, len(ckpt) * height), "white")

    for idx, i in enumerate(ckpt):
        image.paste(Image.open(f'{os.getcwd()}/{project_name}/{model_name}/{i[:-3]}_test.png'), (0, idx * height))

    image.save(f'{project_name}/{model_name}/total_test.png')

    height = 2250
    width = 3000
    image = Image.new("RGB", (2 * width, len(ckpt) * height), "white")
    # 合并各个epoch模型val出来的confusion_matrix
    for idx, i in enumerate(ckpt):
        img = Image.new("RGB", (2 * width, height), "white")
        img.paste(Image.open(f'{os.getcwd()}/runs/{task}/{model_name}--{i[:-3]}/confusion_matrix.png'), (0, 0))
        img.paste(Image.open(f'{os.getcwd()}/runs/{task}/{model_name}--{i[:-3]}/confusion_matrix_normalized.png'),
                  (width, 0))
        image.paste(img, (0, idx * height))
    image.save(f'{project_name}/{model_name}/total_confusion_matrix.png')
