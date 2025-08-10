import argparse
import os
import json
from pathlib import Path

import torch
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from super_gradients.training import Trainer
from super_gradients.training.models import get

def run_inference(model, image_paths, conf_threshold):
    results = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: can't read {img_path}")
            continue

        # preprocess if needed - adapte conforme seu código
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # inferência (adaptar para seu método)
        preds = model.predict(img_rgb, conf=conf_threshold)  # exemplo, ajuste conforme API real
        # preds deve conter: boxes [x1, y1, x2, y2], scores, classes

        # Convertendo predições para formato COCO
        for box, score, cls in zip(preds['boxes'], preds['scores'], preds['classes']):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            results.append({
                "image_id": int(Path(img_path).stem),  # ajuste se seu ID for outro
                "category_id": int(cls),
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })
    return results

def evaluate_coco(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main(args):
    # Carregar modelo
    model = get(args.model, num_classes=args.num_classes, pretrained_weights=None)
    trainer = Trainer()
    trainer.load_checkpoint(checkpoint_path=args.weights, model=model)

    # Listar imagens para inferir
    image_dir = Path(args.source)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    # Rodar inferência
    predictions = run_inference(model, image_paths, args.conf)

    # Salvar predições em formato COCO JSON
    pred_json_path = "predictions.json"
    with open(pred_json_path, "w") as f:
        json.dump(predictions, f)

    # Avaliar usando COCOeval
    evaluate_coco(args.gt_annotations, pred_json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo_nas_m")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="pasta com imagens para inferência")
    parser.add_argument("--gt_annotations", type=str, required=True, help="arquivo JSON COCO das anotações verdadeiras")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--num_classes", type=int, default=1)
    args = parser.parse_args()
    main(args)
