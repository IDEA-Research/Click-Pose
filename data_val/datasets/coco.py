import copy
import os
from collections import OrderedDict, defaultdict
import json_tricks as json
import numpy as np
import xtcocotools
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.bottom_up.bottom_up_coco import BottomUpCocoDataset
from mmcv import Config

@DATASETS.register_module()
class BottomUpCocoDatasetWithCenters(BottomUpCocoDataset):
    """
    Minimally modify original BottomUpCocoDataset so that it doesn't break when setting num_joints == 18
    """
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        assert data_cfg['num_joints'] == 18
        data_cfg['num_joints'] = 17

        dataset_info_cfg = Config.fromfile("config/clickpose/coco_info.py")
        dataset_info = dataset_info_cfg._cfg_dict['dataset_info']

        super().__init__(ann_file, img_prefix, data_cfg, pipeline, dataset_info, test_mode=test_mode)
        self.ann_info['num_joints'] = 18
        self.ann_info['with_center'] = True

    def evaluate(self, *args, **kwargs):
        assert self.ann_info['num_joints'] == 18
        self.ann_info['num_joints'] = 17
        out = super().evaluate(*args, **kwargs)
        self.ann_info['num_joints'] = 18
        return out

    def _get_joints(self, anno):
        self.ann_info['num_joints'] -= 1
        out= super()._get_joints(anno)
        self.ann_info['num_joints'] += 1
        return out

@DATASETS.register_module()
class BottomUpCocoDatasetWithCentersAndBoxes(BottomUpCocoDatasetWithCenters):
    """Modify the class above so that it also uses box annotations, in addition to Keypoint Annotations"""
    def _get_clean_box(self, obj, height, width):
        x, y, w, h = obj['bbox']
        # Top left
        x1 = max(0, x)
        y1 = max(0, y)

        # Bottom right
        x2 = x1 + max(0, w)
        x2 = min(width - 1, x2)

        y2 = y1 + max(0, h)
        y2 = min(height - 1, y2)

        if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
            return np.array([[x1, y1],
                             [x2, y1], # Top-right
                             [x1, y2], # Bottom-Left
                             [x2, y2]
                             ])
        else:
            print(obj['bbox'])
            raise RuntimeError("Image does not have proper box")
    
    def _filter_anno(self, anno):
        ixs_to_keep = []
        for i, obj in enumerate(anno):
            try:
                if 'bbox' not in obj:
                    seg_mask =self.coco.annToMask(anno[i])
                    mask_y, mask_x = np.where(seg_mask)
                    x = min(mask_x)
                    y = min(mask_y)
                    w = max(mask_x) - x
                    h = max(mask_y) - y
                    #print("Old:", anno[i]['bbox'], "New", np.array([x, y, w, h]))
                    anno[i]['bbox'] = np.array([x, y, w, h])
                    anno[i]['area'] = w*h

                valid_bbox = anno[i]['bbox'][-1] > 0 and anno[i]['bbox'][-2]>0
                #valid_kps = (np.array(obj['keypoints']) != 0).any()
                if not valid_bbox:
                    raise RuntimeError
                ixs_to_keep.append(i)
            except:
                pass

        return [anno[i] for i in ixs_to_keep]

    def _get_joints(self, anno):
        """Modify get joints function to also include box corners, and discard objects without proper boxes"""
        anno = self._filter_anno(anno)
        joints = super()._get_joints(anno)
        if anno:
            img_ann = self.coco.loadImgs(anno[0]['image_id'])[0]
            height, width = img_ann['height'], img_ann['width']
            
            assert len(anno) == joints.shape[0]
            boxes = np.zeros((joints.shape[0], 4, 3))
            boxes[..., 2] = 2
            for i, obj in enumerate(anno):
                boxes[i, ..., :2] = self._get_clean_box(obj, height, width)

            joints = np.concatenate((joints, boxes), axis = 1)
        
        return joints