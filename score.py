from ultralytics import YOLO

def main():
    model = YOLO(r"")

    # Validate the model
    metrics = model.val(data=r"", imgsz=160, workers=6,batch=4,
                        conf=0.001, iou=0.6, max_det=300, half=True, device="0", dnn=False, plots=True, rect=False,
                        split='val')

    #fitness=0.1*precision+0.5*recall+0.1*mAP50+0.3*mAP50-95
    precision=metrics.box.p
    recall=metrics.box.r
    mAP50=metrics.box.map50
    mAP50_95=metrics.box.map
    fitness=(0.1*precision)+(0.5*recall)+(0.1*mAP50)+(0.3*mAP50_95)
    print('precision=',precision)
    print('recall=',recall)
    print('mAP50=',mAP50)
    print('mAP50_95=',mAP50_95)

    print(fitness)

if __name__ == '__main__':
    main()