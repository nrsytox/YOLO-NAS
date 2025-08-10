import os
import torch
import yaml
from super_gradients.training import models
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_yaml_config(yaml_file):
    """Carrega o arquivo de configuração YAML."""
    with open(yaml_file) as f:
        return yaml.safe_load(f)

def test_model(
    checkpoint_path: str,
    config_path: str,
    batch_size: int = 4,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    model_arch: str = 'yolo_nas_m',
    output_dir: str = 'outputs'
):
    """
    Teste simplificado do modelo YOLO-NAS
    """
    # Carregar configuração YAML
    config = load_yaml_config(config_path)
    test_images_dir = os.path.join(config['Dir'], config['images']['test'])
    test_annotations_path = os.path.join(config['Dir'], config['labels']['test'])
    num_classes = config['nc']
    class_names = config['names']
    
    # Verificar caminhos
    print(f"Diretório de imagens: {test_images_dir}")
    print(f"Arquivo de anotações: {test_annotations_path}")
    
    # Carregar modelo
    print(f"Carregando modelo {model_arch}...")
    model = models.get(model_arch, num_classes=num_classes, pretrained_weights=None)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()
    
    # Configurar dataset de teste
    print("Configurando dataset...")
    test_dataset = COCOFormatDetectionDataset(
        data_dir=test_images_dir,
        json_annotation_file=test_annotations_path,
        input_dim=(640, 640),
        transforms=[
            {'DetectionPaddedRescale': {'input_dim': (640, 640)}},
            {'DetectionStandardize': {'max_value': 255.0}},
            {'DetectionImagePermute': {}},
        ]
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=test_dataset.collate_fn
    )
    
    # Configurar métricas
    metrics = DetectionMetrics(
        num_cls=num_classes,
        post_prediction_callback=model.get_post_prediction_callback(conf=conf_threshold, iou=iou_threshold),
        normalize_targets=True
    )
    
    # Loop de teste
    print(f"Testando {len(test_dataset)} imagens...")
    for imgs, targets in tqdm(test_loader, desc="Processando"):
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        
        with torch.no_grad():
            preds = model(imgs)
        metrics.update(preds, targets)
    
    # Resultados
    results = metrics.compute()
    print("\nResultados:")
    print(f"Precisão @0.5: {results['precision@0.50']:.4f}")
    print(f"Recall @0.5: {results['recall@0.50']:.4f}")
    print(f"mAP@0.50: {results['mAP@0.50']:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Caminho do checkpoint")
    parser.add_argument("--config", required=True, help="Arquivo de configuração YAML")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default="outputs")
    
    args = parser.parse_args()
    
    test_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        batch_size=args.batch_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        output_dir=args.output_dir
    )
