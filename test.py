import os
import torch
import yaml
from super_gradients.training import models
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from super_gradients.training.utils.detection_utils import DetectionCollateFN

def load_yaml_config(yaml_file):
    """Carrega o arquivo de configuração YAML."""
    with open(yaml_file) as f:
        return yaml.safe_load(f)

class FixedPathCOCODataset(COCODetectionDataset):
    """Dataset com caminhos corrigidos para evitar duplicação de paths"""
    def __init__(self, *args, **kwargs):
        # Remove images_dir se estiver presente para evitar conflito
        if 'images_dir' in kwargs:
            kwargs.pop('images_dir')
        super().__init__(*args, **kwargs)
    
    def _load_image(self, index):
        """Corrige o caminho da imagem para usar diretamente data_dir"""
        img_file = os.path.join(self.data_dir, self.images[index])
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image file not found: {img_file}")
        return super()._load_image(index)

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
    Testa o modelo YOLO-NAS usando configuração YAML e calcula métricas.
    Versão final completamente funcional para super-gradients 3.1.3
    """
    # Carregar configuração YAML
    config = load_yaml_config(config_path)
    test_images_dir = os.path.join(config['Dir'], config['images']['test'])
    test_annotations_path = os.path.join(config['Dir'], config['labels']['test'])
    num_classes = config['nc']
    class_names = config['names']
    
    # Verificar caminhos
    print("\nVerificando caminhos...")
    print(f"Diretório de imagens: {test_images_dir}")
    print(f"Arquivo de anotações: {test_annotations_path}")
    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {test_images_dir}")
    if not os.path.exists(test_annotations_path):
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {test_annotations_path}")
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar modelo
    print(f"\nCarregando modelo {model_arch} com {num_classes} classes...")
    model = models.get(model_arch, num_classes=num_classes, pretrained_weights=None)
    
    # Carregar checkpoint
    print(f"Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()
    
    # Configurar dataset de teste com nossa classe corrigida
    print("\nConfigurando dataset de teste...")
    test_dataset = FixedPathCOCODataset(
        data_dir=test_images_dir,  # Caminho direto para as imagens
        json_file=test_annotations_path,
        input_dim=(640, 640),
        transforms=[
            {'DetectionPaddedRescale': {'input_dim': (640, 640)}},
            {'DetectionStandardize': {'max_value': 255.0}},
            {'DetectionImagePermute': {}},
        ]
    )
    
    # Verificar algumas imagens
    print("\nVerificando acesso às imagens...")
    for i in range(min(3, len(test_dataset))):
        try:
            img, _ = test_dataset[i]
            print(f"Imagem {i} carregada com sucesso")
        except Exception as e:
            print(f"Erro ao carregar imagem {i}: {str(e)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=DetectionCollateFN()
    )
    
    # Configurar métricas
    print("\nConfigurando métricas de avaliação...")
    metrics = DetectionMetrics(
        num_cls=num_classes,
        post_prediction_callback=model.get_post_prediction_callback(conf=conf_threshold, iou=iou_threshold),
        normalize_targets=True
    )
    
    # Loop de teste
    print(f"\nIniciando teste em {len(test_dataset)} imagens...")
    progress_bar = tqdm(test_loader, desc="Processando batches")
    
    try:
        for batch_idx, (imgs, targets) in enumerate(progress_bar):
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets]
            
            with torch.no_grad():
                preds = model(imgs)
            
            metrics.update(preds, targets)
    except Exception as e:
        print(f"\nErro durante o teste: {str(e)}")
        raise
    
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
    
    parser = argparse.ArgumentParser(description="Teste do YOLO-NAS com YAML - Versão Final Funcional")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho para o checkpoint .pth")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o arquivo YAML de configuração")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch para teste")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Limiar de confiança")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="Limiar de IoU para NMS")
    parser.add_argument("--model_arch", type=str, default="yolo_nas_m", 
                        choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"], help="Arquitetura do modelo")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Pasta de saída")
    
    args = parser.parse_args()
    
    results = test_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        batch_size=args.batch_size,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        model_arch=args.model_arch,
        output_dir=args.output_dir
    )
