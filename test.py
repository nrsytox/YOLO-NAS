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

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    num_classes = len(data['names'])
    # Criar o Trainer
    trainer = Trainer(experiment_name='yolo_nas_test', ckpt_root_dir='runs')

    # Carregar o modelo
    model = get('yolo_nas_m', num_classes=num_classes, pretrained_weights=None)
    model = model.to(device)

    # Carregar o checkpoint
    trainer.load_checkpoint(checkpoint_path=weight_path, model=model)

    # Preparar o dataset e o dataloader
    testset = COCOFormatDetectionDataset(
        data_dir=data_yaml['Dir'],
        images_dir=data_yaml['images']['test'],
        json_annotation_file=data_yaml['labels']['test'],
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

    # Rodar o teste
    results = trainer.test(model=model, test_loader=test_loader, device=device, confidence_threshold=confidence_threshold)

    # Exibir os resultados
    print("\n===== RESULTADOS =====")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"mAP@0.50:  {results['mAP@0.50']:.4f}")
    if 'mAP@0.50:0.95' in results:
        print(f"mAP@0.50:0.95: {results['mAP@0.50:0.95']:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python test.py <caminho_data_yaml> <caminho_pesos>")
        sys.exit(1)
    data_yaml = sys.argv[1]
    weight_path = sys.argv[2]
    main(data_yaml, weight_path)
