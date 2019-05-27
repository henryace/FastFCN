from .base import *
from .coco import COCOSegmentation
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CitySegmentation
from .custom import CustomSegmentation
from .custom import CustomSegmentation
from .custom import CustomSegmentation
from .custom import CustomSegmentation

# 
# add : CustomSegmentation
# 20190524 : add tpwdataset
# CMH
#

datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'citys': CitySegmentation,
    'custom': CustomSegmentation,
    'tpw_dataset_1': TpwDataset1,
    'tpw_dataset_1': TpwDataset2,
    'tpw_dataset_1': TpwDataset3,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
