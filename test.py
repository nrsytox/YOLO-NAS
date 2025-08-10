import sys
import yaml
import torch
from super_gradients.training import Trainer
from super_gradients.training.models import get
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from super_gradients.training import dataloaders
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN

def main(data_yaml, weight_path, batch_size=4, confidence_threshold=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(experiment_name='yolo_nas_test', ckpt_root_dir='runs')

    # Carregar modelo
    model = get('yolo_nas_m', num_classes=1, pretrained_weights=None).to(device)

    # Carregar checkpoint para o modelo
    trainer.load_checkpoint(checkpoint_path=weight_path, model=model)

    # Preparar dataset e dataloader para teste
    import yaml
    with open(data_yaml, 'r') as f:
        yaml_params = yaml.safe_load(f)

    testset = COCOFormatDetectionDataset(
        data_dir=yaml_params['Dir'],
        images_dir=yaml_params['images']['val'],  # ajustar se seu yaml usa 'val' ou 'test'
        json_annotation_file=yaml_params['labels']['val'],
        input_dim=(640, 640),
        ignore_empty_annotations=False,
        transforms=[
            DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
            DetectionStandardize(max_value=255),
            DetectionTargetsFormatTransform(max_targets=300, input_dim=(640, 640), output_format="LABEL_CXCYWH")
        ]
    )

    test_loader = dataloaders.get(dataset=testset, dataloader_params={
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": 2,
        "drop_last": False,
        "pin_memory": True,
        "collate_fn": CrowdDetectionCollateFN(),
        "worker_init_fn": None
    })

    # Rodar teste
    results = trainer.test(model=model, test_loader=test_loader, device=device, confidence_threshold=confidence_threshold)

    # Mostrar métricas
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
