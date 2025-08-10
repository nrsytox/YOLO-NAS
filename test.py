import sys
import yaml
import torch
from super_gradients.training import Trainer
from super_gradients.training.models import get

def main(data_yaml, weight_path, model_name='yolo_nas_m', batch_size=4, confidence_threshold=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ler número de classes do YAML
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
        num_classes = len(data['names'])

    # Criar trainer
    trainer = Trainer(experiment_name='yolo_nas_test', ckpt_root_dir='runs')

    # Carregar modelo com número de classes correto
    model = get(model_name, num_classes=num_classes, pretrained_weights=None)
    model = model.to(device)

    # Carregar checkpoint no modelo
    trainer.load_checkpoint_to_model(model, weight_path, load_backbone=False, strict=False)

    # Rodar teste — usa conjunto de validação definido no YAML
    results = trainer.test(
        model=model,
        test_loader=data_yaml,
        device=device,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold
    )

    # Exibir métricas principais
    print("\n===== RESULTADOS =====")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"mAP@0.50:  {results['mAP@0.50']:.4f}")
    if 'mAP@0.50:0.95' in results:
        print(f"mAP@0.50:0.95: {results['mAP@0.50:0.95']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python test.py <data_yaml> <weight_path>")
        sys.exit(1)
    _, data_yaml, weight_path = sys.argv
    main(data_yaml, weight_path)
