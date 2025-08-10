import argparse
import torch
import yaml
from super_gradients.training.models import get
from supervision import DetectionMetrics

def load_model(weight_path, num_classes, device):
    model = get('yolo_nas_m', num_classes=num_classes, pretrained_weights=None)
    model = model.to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def main(data_yaml, weights_list, batch_size, confidence_threshold):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    num_classes = len(data['names'])

    # Aqui precisa criar o test_loader com seu dataset, ex:
    # test_loader = ...

    for weight_path in weights_list:
        print(f"\n=== Testando weight: {weight_path} ===")
        model = load_model(weight_path, num_classes, device)
        metrics = DetectionMetrics(iou_threshold=0.5)

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                predictions = model(images)
                metrics.update(predictions, targets)

        print(f"Precision@{confidence_threshold}: {metrics.precision():.4f}")
        print(f"Recall@{confidence_threshold}: {metrics.recall():.4f}")
        print(f"mAP@{confidence_threshold}: {metrics.mean_average_precision():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testar YOLO-NAS com múltiplos pesos")
    parser.add_argument("--data", type=str, required=True, help="Caminho para o arquivo data.yaml")
    parser.add_argument("--weights", nargs="+", required=True, help="Lista de pesos (.pth) para testar")
    parser.add_argument("--batch", type=int, default=4, help="Batch size para teste")
    parser.add_argument("--conf", type=float, default=0.5, help="Threshold de confiança")

    args = parser.parse_args()
    main(args.data, args.weights, args.batch, args.conf)
