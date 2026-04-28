from .datasets import *
from .models import *
from .apis import *
from .core.evaluation import *
from .mmcv_custom import *
from .datasets import *
from .models import *
from .apis import *
from .core.evaluation import *
from .mmcv_custom import *

# ===== Register mmdet3d point-cloud pipelines into mmdet PIPELINES =====
# Sparse4D uses mmdet.datasets.pipelines.Compose, so mmdet3d pipeline modules
# must be explicitly registered into mmdet PIPELINES.
from mmdet.datasets.builder import PIPELINES

from mmdet3d.datasets.pipelines.loading import (
    LoadPointsFromFile,
    LoadPointsFromMultiSweeps,
)

from mmdet3d.datasets.pipelines.transforms_3d import (
    PointsRangeFilter,
    PointShuffle,
)

PIPELINES.register_module(module=LoadPointsFromFile, force=True)
PIPELINES.register_module(module=LoadPointsFromMultiSweeps, force=True)
PIPELINES.register_module(module=PointsRangeFilter, force=True)
PIPELINES.register_module(module=PointShuffle, force=True)