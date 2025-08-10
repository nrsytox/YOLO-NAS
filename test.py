import sys
from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
from super_gradients.training.models.model_factory import instantiate_model
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.utils.callback_utils import PhaseCallback, PhaseContext
import torch

def main(data_yaml, weight_path, model_name='yolo_nas_m', batch_size=4):
    # Configurar o dataset de validação baseado no YAML
    val_dataset = COCOFormatDetectionDataset(
        data_dir=data_yaml,
        img_size=None,  # ajuste conforme seu setup
        split='val'
    )
    val_loader = val_dataset.to_loader(
        shuffle=False, batch_size=batch_size, drop_last=False
    )

    # Instanciar o modelo
    model = instantiate_model(model_name, num_classes=val_dataset.num_classes, pretrained_weights=None)
    model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(model_device)

    # Carregar pesos
    checkpoint = load_checkpoint_to_model(weight_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Criar Callback para métricas
    metric = DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        num_cls=val_dataset.num_classes
    )
    metric_cb = PhaseCallback(callback_fn=lambda context: metric.update(context.data['predictions'], context.data['targets']))

    # Treinar inválido apenas aplicar avaliação
    trainer = Trainer(experiment_name='test', ckpt_root_dir='runs', resume=False)
    trainer.model = model
    trainer._results_callback_list = [metric_cb]

    # Rodar avaliação
    print("Iniciando avaliação no conjunto de validação...")
    trainer._run_phase('valid', val_loader)

    # Imprimir métricas
    prec, rec, mAP50 = metric.compute()
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"mAP@0.50:  {mAP50:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python test.py <data_yaml> <weight_path>")
        sys.exit(1)
    _, data_yaml, weight_path = sys.argv
    main(data_yaml, weight_path)
