import copy
import cv2
import numpy as np
import torch
from mmpose.core.post_processing import get_affine_transform
from mmpose.datasets.builder import PIPELINES
from mmpose.datasets.pipelines.bottom_up_transform import BottomUpRandomFlip, _resize_align_multi_scale, Compose, BottomUpRandomAffine

def _get_pad_mask(original_img, center, scale, size_resized):
    pad_mask = (np.ones_like(original_img) * 255).astype(np.uint8)
    trans = get_affine_transform(center, scale, 0, size_resized)
    pad_mask = cv2.warpAffine(pad_mask, trans, size_resized)
    pad_mask = (pad_mask/255 > 0.5).astype(np.float32)[..., 0]
    return pad_mask

@PIPELINES.register_module()
class GenerateRootNode:
    def __init__(self, from_box=False):
        self.from_box= from_box

    def __call__(self, results):     
        if len(results['joints'][0]):
            assert results['joints'][0].shape[1] == results['ann_info']['num_joints'] - 1   
        
        joints_ = results['joints']
        for i, (joints, max_size) in enumerate(zip(joints_, results['ann_info']['heatmap_size'])):
            if not self.from_box:
                in_range_x = (joints[...,0] >= 0) & (joints[...,1] >= 0)
                in_range_y = (joints[...,0] < max_size) & (joints[...,1] < max_size)
                was_vis = joints[..., 2] > 0
                is_vis = was_vis & in_range_x & in_range_y
                
                centers = (joints*is_vis[..., None].astype(float)).sum(-2) / (is_vis.sum(-1)[..., None] + 1e-9)
                
                centers[..., 2] = 2*(is_vis.sum(-1) > 0).astype(float) # 2 if there's at least one visible node, 0 otherwise

            else:
                boxes = results['boxes'][i] 
                assert boxes.shape[0] == joints.shape[0]
                assert (boxes[..., 2].max(-1) == boxes[..., 2].min(-1)).all(), "Some boxes corners are labelled as visible and some are not"
                centers = boxes.mean(-2)
                #raise NotImplementedError
            
            joints_[i] = np.concatenate((joints_[i], centers[:, None]), axis = 1)

        results['joints'] = joints_

        return results

@PIPELINES.register_module()
class BottomUpResizeAlignWPadMask:
    """Resize multi-scale size and align transform for bottom-up.
    Args:
        transforms (List): ToTensor & Normalize
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, transforms, use_udp=False):
        self.transforms = Compose(transforms)
        if use_udp:
            #self._resize_align_multi_scale = _resize_align_multi_scale_udp
            raise RuntimeError("UDP is not supported")
        else:
            self._resize_align_multi_scale = _resize_align_multi_scale
    
    def _do_asserts(self, results):
        assert results['ann_info']['test_scale_factor'][1] == 1
        assert isinstance (results['pad_mask'], list)
        assert len(results['ann_info']['aug_data']) == len(results['ann_info']['test_scale_factor'])
        assert isinstance(results['ann_info']['aug_data'][1], torch.Tensor)
        assert results['ann_info']['aug_data'][1].dim()==4 and results['ann_info']['aug_data'][1].shape[0]==1


    def __call__(self, results):
        """Resize multi-scale size and align transform for bottom-up."""
        input_size = results['ann_info']['image_size']
        test_scale_factor = results['ann_info']['test_scale_factor']
        aug_data = []
        assert 1 in test_scale_factor
        for _, s in enumerate(sorted(test_scale_factor, reverse=True)):
            _results = results.copy()
            image_resized, center, scale = self._resize_align_multi_scale(
                _results['img'], input_size, s, min(test_scale_factor))

            if s == 1:
                h, w, _ =  image_resized.shape                
                pad_masks = [_get_pad_mask(_results['img'], center, scale, (w//4, h//4)),
                             _get_pad_mask(_results['img'], center, scale, (w//2, h//2)),
                             _get_pad_mask(_results['img'], center, scale, (w, h))]

            _results['img'] = image_resized
            _results = self.transforms(_results)
            transformed_img = _results['img'].unsqueeze(0)
            aug_data.append(transformed_img)

        results['ann_info']['aug_data'] = aug_data
        results['pad_mask']=pad_masks
        results['img'] = results['ann_info']['aug_data'][1][0]
        self._do_asserts(results)

        return results
        
@PIPELINES.register_module()
class AddHasKPAnns:
    """ 
        Determines whether each person in the image has kp annotations or not
        Needs to be called before BottomUpGenerateTarget, otherwise Joints will be modified, and also before AddAndUpdateVisibility.
        It can be called as the first transform. 
    """
    def __init__(self):
        pass
    def __call__(self, results):
        joints = results['joints'][0]
        results['has_kp_anns'] = (joints[..., 2] !=0).max(axis = -1)
        
        return results

@PIPELINES.register_module()
class AddAndUpdateVisibility:
    """ 
    Determines which persons still have visible joints after data augmentation. Note that with cropping, etc.
    some poses might completely disappear from the scene.
    Needs to be called before BottomUpGenerateTarget, otherwise Joints will be modified"""
    def __init__(self):
        pass
        
    def __call__(self, results):     
        if len(results['joints'][0]):
            assert results['joints'][0].shape[1] == results['ann_info']['num_joints'] - 1   
        
        joints = results['joints'][-1]
        max_size = results['ann_info']['heatmap_size'][-1]

        in_range_x = (joints[...,0] >= 0) & (joints[...,1] >= 0)
        in_range_y = (joints[...,0] < max_size) & (joints[...,1] < max_size)
        was_vis = joints[..., 2] > 0
        kp_visibility = (was_vis & in_range_x & in_range_y)
        
        for i in range(len(results['joints'])):
            results['joints'][i][..., 2] = kp_visibility

        results['obj_vis'] = kp_visibility.max(-1) > 0
                
        return results

@PIPELINES.register_module()
class CloneJoints:
    """Clone Joints before BottomUpGenerateTarget messes them up"""
    def __init__(self):
        pass

    def __call__(self, results):     
        results['joints_'] = copy.deepcopy(results['joints'])
        return results


@PIPELINES.register_module()
class PadArrays:
    def __init__(self, keys_to_update, max_num_people):
        self.keys_to_update = keys_to_update
        self.max_num_people = max_num_people
    
    def pad_to_max_num_persons(self, array):
        assert len(array.shape) >= 2
        if len(array.shape) == 2: # 2nd Dimensions is nb of 'keypoint types'
            array_ = array[:, None].copy()
            extra_dim=True

        else:
            array_ = array.copy()
            extra_dim=False

        num_people = min(array.shape[0], self.max_num_people)
        pad_array = np.zeros([self.max_num_people] + [val for val in list(array_.shape)[1:]])
        pad_array[:num_people] = array_[:num_people]

        if extra_dim:
            return pad_array[:, 0]
        
        else:
            return pad_array

    def __call__(self, results):
        for i, _output_size in enumerate(results['ann_info']['heatmap_size']):
            for key in self.keys_to_update:
                if key in results:
                    if isinstance(results[key], (tuple, list)) and i < len(results[key]):
                        results[key][i] = self.pad_to_max_num_persons(results[key][i])
                    
                    elif isinstance(results[key], np.ndarray) and i == 0:
                        #print(f"UPDATING KEY {key}")
                        array = results[key]
                        assert len(array.shape) == 1
                        num_people = min(array.shape[0], self.max_num_people)
                        pad_array = np.zeros(self.max_num_people, dtype = array.dtype)
                        pad_array[:num_people] = array[:num_people]
                        results[key] = pad_array

        return results


@PIPELINES.register_module()
class BottomUpRandomAffineWPadMask(BottomUpRandomAffine):
    """Hacky way to inherit from BottomUpRandomAffine so that it also produces Padding Masks
    Padding Masks == binary mask indicating which region in the augmented image was not present in the original image
    We need it for Positional Encodings.
    """
    def __call__(self, results):
        # Add a ones mask to 'ignore masks', that we will used to obtain a mask highlighting the padded areas
        assert len(results['mask'][0].shape) == 2
        results['mask'] = [np.concatenate((mask[..., None], np.ones_like(mask)[..., None]), axis=-1) for mask in results['mask']]
        results = super().__call__(results)
    
        # Split masks into ignore and padding masks
        masks = results['mask'] 
        results['pad_mask'] = [mask[..., 1].copy() for mask in masks]
        results['mask'] = [mask[..., 0].copy() for mask in masks]

        return results

@PIPELINES.register_module()
class BottomUpRandomFlipWPadMaskAndBoxes(BottomUpRandomFlip):
    def __call__(self, results):

        # Merge pad masks and ignore masks again
        results['mask'] = [np.concatenate((mask[..., None], pad_mask[..., None]), axis=-1) for mask, pad_mask in zip(results['mask'], results['pad_mask'])]

        # Modify flip_index
        flip_index = results['ann_info']['flip_index']
        num_joints = len(flip_index)
        box_flip_index = num_joints + np.array([1, 0, 3, 2])
        results['ann_info']['flip_index'] = np.concatenate([flip_index, box_flip_index])
        
        if results['joints']:
            assert results['joints'][0].shape[1] == num_joints + 4, "Box coordinates seem to not be contained inside the joints field"

        # Modify back flip_index
        results = super().__call__(results)
        results['ann_info']['flip_index'] = results['ann_info']['flip_index'][:-4]
        
        # Modify back masks 
        results['pad_mask'] = [mask[..., 1].copy() for mask in results['mask']]
        results['mask'] = [mask[..., 0].copy() for mask in results['mask']]

        return results

@PIPELINES.register_module()
class SeparateJointsAndBoxes:
    """Get boxes in a separate entry inside results, as we initially place them as additional keypoints to reuse code"""
    def __call__(self, results):        
        results['boxes'] = [joints_[:, -4:][:, [0, 3]].copy() for joints_ in results['joints']]
        #results['joints'] = [joints_[:, :-4].copy() for joints_ in results['joints']]
        results['joints'] = [joints_[:, :-4].copy() for joints_ in results['joints']]

        return results