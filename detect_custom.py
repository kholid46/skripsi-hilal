import torch
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.datasets import LoadImages
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
import os
import pandas as pd
import cv2

def run_detection(source, weights='best.pt', img_size=640, conf_thres=0.25):
    device = select_device('')  # otomatis pilih CPU/GPU
    model = DetectMultiBackend(weights, device=device)
    model.eval()
    
    dataset = LoadImages(source, img_size=img_size)
    names = model.names
    result_img_path = "output/detected.jpg"
    result_csv_path = "output/result.csv"
    
    rows = []
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        img = img.unsqueeze(0) if img.ndimension() == 3 else img

        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, 0.45)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    xywh = xyxy2xywh(torch.tensor([xyxy]))[0].tolist()
                    rows.append({
                        'label': names[int(cls)],
                        'confidence': float(conf),
                        'x': xywh[0],
                        'y': xywh[1],
                        'width': xywh[2],
                        'height': xywh[3]
                    })
                    cv2.rectangle(im0s, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(im0s, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imwrite(result_img_path, im0s)

    df = pd.DataFrame(rows)
    df.to_csv(result_csv_path, index=False)

    return result_img_path, result_csv_path
