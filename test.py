import sys
import yaml
import torch
from super_gradients.training import Trainer
from super_gradients.training.models import get
from super_gradients.training.datasets.detection_datasets.coco_format import COCOFormatDetectionDataset
from super_gradients.training.transforms.detection_transforms import (
    DetectionPaddedRescale,
    DetectionStandardize,
    DetectionTargetsFormatTransform
)
from super_gradients.training.datasets.dataloaders import dataloaders
from super_gradients.training.datasets.detection_datasets.utils import CrowdDetectionCollateFN
from super_gradients.training.datasets.detection_datasets.utils import worker_init_reset_seed

def main(data_yaml, weight_path, model_name='yolo_nas_m', batch_size=4, confidence_threshold=0.5, size=640, num_workers=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Carregar YAML
    with open(data_yaml) as f:
        yaml_params = yaml.safe_load(f)

    # Criar trainer
    trainer = Trainer(experiment_name='yolo_nas_test', ckpt_root_dir='runs')

    # Número de classes
    num_classes = len(yaml_params['names'])

    # Criar modelo
    model = get(model_name, num_classes=num_classes, pretrained_weights=None)
    model = model.to(device)

    # Carregar checkpoint no modelo
    trainer._load_checkpoint_to_model(model, weight_path, strict=False)

    # Criar test dataset e loader
    if 'test' in (yaml_params['images'].keys() or yaml_params['labels'].keys()):
        testset = COCOFormatDetectionDataset(
            data_dir=yaml_params.get('Dir', ''),
            images_dir=yaml_params['images']['test'],
            json_annotation_file=yaml_params['labels']['test'],
            input_dim=(size, size),
            ignore_empty_annotations=False,
            transforms=[
                DetectionPaddedRescale(input_dim=(size, size), max_targets=300),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(max_targets=300, input_dim=(size, size), output_format="LABEL_CXCYWH")
            ]
        )
        test_loader = dataloaders.get(dataset=testset, dataloader_params={
            "shuffle": False,
            "batch_size": batch_size * 2,
            "num_workers": num_workers,
            "drop_last": False,
            "pin_memory": True,
            "collate_fn": CrowdDetectionCollateFN(),
            "worker_init_fn": worker_init_reset_seed
        })
    else:
        print("Dataset de teste não encontrado no YAML!")
        return

    # Rodar teste
    results = trainer.test(
        model=model,
        test_loader=test_loader,
        device=device,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold
    )

    print("\n===== RESULTADOS =====")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"mAP@0.50:  {results['mAP@0.50']:.4f}")
    if 'mAP@0.50:0.95' in results:
        print(f"mAP@0.50:0.95: {results['mAP@0.50:0.95']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python test.py <data_yaml> <weight_path> [batch_size] [confidence_threshold]")
        sys.exit(1)
    data_yaml = sys.argv[1]
    weight_path = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    confidence_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    main(data_yaml, weight_path, batch_size=batch_size, confidence_threshold=confidence_threshold)
