"""Install: pip install .

Examples:
  python train.py --list-models
  python train.py --data /your/dataset.yaml --model yolo26-eca --device 0
  python train.py --data /your/dataset.yaml --model yolo26 --weights /your/yolo26n.pt

Dataset descriptions and trusted weights are supplied by the user, never bundled.
This minimal entry point supports one GPU or CPU; distributed training is excluded.
"""
import argparse
from pathlib import Path

from underwater26 import ARCHITECTURES, build_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--model', choices=sorted(ARCHITECTURES), default='yolo26')
    parser.add_argument('--scale', choices=['n', 's', 'm', 'l', 'x'], default='n')
    parser.add_argument('--data', help='Path to your own dataset description')
    parser.add_argument('--weights', help='Optional trusted local checkpoint')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default='runs/underwater26')
    parser.add_argument('--name', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.list_models:
        print('\n'.join(sorted(ARCHITECTURES)))
        return 0
    if not args.data or not Path(args.data).expanduser().is_file():
        parser.error('--data must point to your own existing dataset description')
    if ',' in args.device:
        parser.error('This minimal entry point supports a single GPU or CPU')
    if args.weights and not Path(args.weights).expanduser().is_file():
        parser.error('--weights must point to an existing trusted local checkpoint')
    model = build_model(args.model, args.scale, weights=Path(args.weights).expanduser() if args.weights else None)
    model.train(
        data=str(Path(args.data).expanduser().resolve()),
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=args.workers, project=args.project,
        name=args.name or args.model, seed=args.seed,
        optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=0.3, mixup=0.1, copy_paste=0.0, close_mosaic=15,
        patience=50, pretrained=bool(args.weights), deterministic=True,
        plots=False, resume=False,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
