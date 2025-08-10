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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    model_arch: str = 'yolo_nas_m',
    visualize: bool = False,
    output_dir: str = 'outputs'
):
    """
    Testa o modelo YOLO-NAS usando configuração YAML e calcula métricas.
    """
    # Carregar configuração YAML
    config = load_yaml_config(config_path)
    test_images_dir = os.path.join(config['Dir'], config['images']['test'])
    test_annotations_path = os.path.join(config['Dir'], config['labels']['test'])
    num_classes = config['nc']
    class_names = config['names']
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar modelo
    print(f"Carregando modelo {model_arch} com {num_classes} classes...")
    model = models.get(model_arch, num_classes=num_classes, checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Obter tamanho de entrada do modelo (ajuste para versões mais recentes do SuperGradients)
    try:
        input_dim = model._image_size
    except AttributeError:
        try:
            input_dim = model._default_nms_conf.image_size
        except AttributeError:
            input_dim = [640, 640]  # Valor padrão se não for encontrado
    
    # Configurar dataset de teste
    print("Configurando dataset de teste...")
    test_dataset = COCOFormatDetectionDataset(
        data_dir=test_images_dir,
        json_annotation_file=test_annotations_path,
        input_dim=input_dim,
        transforms=model._preprocessing_transforms
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
        post_prediction_callback=model._get_post_prediction_callback(conf=conf_threshold),
        normalize_targets=True,
        calc_best_score=False
    )
    
    # Loop de teste
    print(f"Iniciando teste em {len(test_dataset)} imagens...")
    progress_bar = tqdm(test_loader, desc="Processando batches")
    
    for batch_idx, (imgs, targets) in enumerate(progress_bar):
        imgs = imgs.to(device)
        
        with torch.no_grad():
            preds = model(imgs)
        
        # Converter predições para formato de métricas
        formatted_preds = []
        for img_idx, detections in enumerate(preds):
            if len(detections.prediction) == 0:
                formatted_preds.append(torch.zeros((0, 6), device=device))
                continue
            
            # Formatar como [x1, y1, x2, y2, conf, class]
            boxes = detections.prediction[:, :4]
            scores = detections.prediction[:, 4]
            classes = detections.prediction[:, 5]
            
            formatted_pred = torch.cat([
                boxes,
                scores.unsqueeze(1),
                classes.unsqueeze(1)
            ], dim=1)
            formatted_preds.append(formatted_pred)
        
        # Atualizar métricas
        metrics.update(formatted_preds, targets)
    
    # Calcular métricas finais
    print("\nCalculando métricas finais...")
    metrics_results = metrics.compute()
    
    # Resultados
    print("\n=== Resultados do Teste ===")
    print(f"Precisão @0.5 (P0.5): {metrics_results['precision@0.50']:.4f}")
    print(f"Recall @0.5 (R0.5): {metrics_results['recall@0.50']:.4f}")
    print(f"mAP@0.50: {metrics_results['mAP@0.50']:.4f}")
    
    return metrics_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Teste do YOLO-NAS com YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho para o checkpoint .pth")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo YAML de configuração")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch para teste")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Limiar de confiança")
    parser.add_argument("--model_arch", type=str, default="yolo_nas_m", 
                        choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"], help="Arquitetura do modelo")
    parser.add_argument("--visualize", action="store_true", help="Gerar visualizações")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Pasta de saída")
    
    args = parser.parse_args()
    
    results = test_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        batch_size=args.batch_size,
        conf_threshold=args.conf_threshold,
        model_arch=args.model_arch,
        visualize=args.visualize,
        output_dir=args.output_dir
    )
