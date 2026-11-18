"""Underwater YOLO26 extensions extracted from the supplied research project."""

from .models import ARCHITECTURES, model_definition

__all__ = ['ARCHITECTURES', 'model_definition', 'enable_extensions', 'build_model']
_enabled = False


def enable_extensions():
    """Register the project's parser, layers and active GIoU loss in this process."""
    global _enabled
    if _enabled:
        return
    import ultralytics
    if ultralytics.__version__ != '8.4.48':
        raise RuntimeError('This extension requires ultralytics==8.4.48')
    from ultralytics.nn import tasks, modules
    from ultralytics.utils import loss, metrics, tal
    from .layers import CUSTOM_LAYERS
    from .parser import parse_model
    from .overlaps import bbox_iou
    from .loss import BboxLoss
    for name, cls in CUSTOM_LAYERS.items():
        setattr(modules, name, cls)
        setattr(tasks, name, cls)
    tasks.parse_model = parse_model
    metrics.bbox_iou = loss.bbox_iou = tal.bbox_iou = bbox_iou
    loss.BboxLoss = BboxLoss
    _enabled = True


def build_model(name='yolo26', scale='n', classes=80, weights=None, verbose=False):
    """Build a model from its Python graph; optionally load a trusted checkpoint."""
    enable_extensions()
    from .model import UnderwaterYOLO
    model = UnderwaterYOLO(model_definition(name, scale, classes), verbose=verbose)
    if weights:
        model.load(str(weights))
    return model
