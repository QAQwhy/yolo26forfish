# Derived from Ultralytics AGPL-3.0; see LICENSE.
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT


class UnderwaterYOLO(YOLO):
    """Use an in-memory graph instead of a model configuration file."""

    def __init__(self, definition, verbose=False):
        object.__setattr__(self, '_definition', definition)
        # This identifier selects _new(); no YAML file is opened or generated.
        super().__init__('underwater26.yaml', task='detect', verbose=verbose)

    def _new(self, cfg, task=None, model=None, verbose=False):
        self.cfg = cfg
        self.task = 'detect'
        self.model = DetectionModel(self._definition, verbose=verbose)
        self.overrides.update(model=cfg, task=self.task)
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg
