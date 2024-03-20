import time
from ultralytics import YOLO
import os
from PIL import Image

project_name = 'yolov8'
dataset_name = '456.yaml'
height = 1500
width = 2250
imgsz = 480
optimizer = 'Adam'
model_name = f'456_300dataset_imgsz{imgsz}_v8n_{optimizer}'

batchsz = -1

if __name__ == '__main__':
    # train and val

    model = YOLO('yolov8n.yaml')
    # model = YOLO('yolov8/123_imgsz1280_v8n_Adam/weights/last.pt')

    model.train(data=dataset_name, epochs=1500,
                imgsz=imgsz, batch=batchsz, cache='disk',
                pretrained=False, optimizer=optimizer,
                amp=True, fliplr=0,
                val=True, save_period=50, patience=100,
                name=model_name, project=project_name,
                verbose=True)

    # model.train(resume=True)
    del model
    time.sleep(60)

    # test

    ckpt = os.listdir(f'{project_name}/{model_name}/weights')

    ckpt.remove('best.pt')
    ckpt.remove('last.pt')
    ckpt.sort(key=lambda x: int(x[5:][:-3]))
    ckpt.extend(['last.pt', 'best.pt'])

    metrics = ['F1_curve.png', 'P_curve.png', 'PR_curve.png', 'R_curve.png']

    for id1, i in enumerate(ckpt):

        model = YOLO(f'{project_name}/{model_name}/weights/{i}')
        model.val(data=dataset_name, split='test', imgsz=imgsz, batch=16, half=True, plots=True,
                  name=f'{model_name}--{i[:-3]}')
        del model
        # time.sleep(60)

        image = Image.new("RGB", (len(metrics) * width, height), "white")

        for id2, j in enumerate(metrics):
            image.paste(Image.open(f'{os.getcwd()}/runs/detect/{model_name}--{i[:-3]}/{j}'), (id2 * width, 0))

        image.save(f'{project_name}/{model_name}/{i[:-3]}_test.png')

    image = Image.new("RGB", (len(metrics) * width, len(ckpt) * height), "white")

    for idx, i in enumerate(ckpt):
        image.paste(Image.open(f'{os.getcwd()}/{project_name}/{model_name}/{i[:-3]}_test.png'), (0, idx * height))

    image.save(f'{project_name}/{model_name}/total_test.png')
