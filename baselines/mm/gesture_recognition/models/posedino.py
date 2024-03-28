from typing import Union, Tuple

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.models.detectors import DINO
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor

from mmpose.registry import MODELS as POSE_MODELS

from torch import Tensor
from torch.nn import Conv2d

@MODELS.register_module()
class POSEDINO(DINO):
    def __init__(self,
                 *args,
                 pose_heads: OptConfigType,
                 pose_neck: OptConfigType,
                 pose_weight: float=1.,
                 det_weight: float=1.,
                 **kwargs):
        super().__init__(*args,**kwargs)
        self.pose_weight = pose_weight
        self.det_weight = det_weight
        self.pose_heads = POSE_MODELS.build(pose_heads)
        self.pose_neck = POSE_MODELS.build(pose_neck)

    def forward(self,
                inputs: Tensor,
                det_samples: OptSampleList,
                pose_samples: OptSampleList,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.loss(inputs, det_samples, pose_samples)
        elif mode == 'predict':
            return self.predict(inputs, det_samples, pose_samples)
        elif mode == 'tensor':
            return self._forward(inputs, det_samples, pose_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')


    def loss(self, batch_inputs: Tensor,
             det_samples,
             pose_samples) -> Union[dict, list]:
        det_feats, pose_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(det_feats,
                                                    det_samples)
        det_losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=det_samples)
        
        pose_losses = self.pose_heads.loss(pose_feats, pose_samples)

        det_losses = {k:v*self.det_weight for k,v in det_losses.items()}
        pose_losses = {k:v*self.pose_weight for k,v in pose_losses.items()}


        return det_losses | pose_losses


    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            det_feats = self.neck(x)
        pose_feats = self.pose_neck(x)
        return det_feats, pose_feats
    
    def predict(self,
                batch_inputs: Tensor,
                det_data_samples,
                pose_data_samples,
                rescale: bool = True):
        img_feats,_ = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    det_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=det_data_samples)

        # todo add stuff here somewhere
        det_data_samples = self.add_pred_to_datasample(
            det_data_samples, results_list)
        return det_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            det_data_samples,
            pose_data_samples):
        det_feats, pose_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(det_feats,
                                                    det_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results


@MODELS.register_module()
class MultiTaskDataPreprocessor(DetDataPreprocessor):
    def __init__(self, pose_preprocessor, **kwargs):
        super().__init__(**kwargs)
        self.pose_preprocessor = POSE_MODELS.build(pose_preprocessor)

    def forward(self, data: dict, training: bool = False) -> dict:
        det_data = {'inputs': data['inputs'], 'data_samples': data['det_samples']}
        det_processed = super().forward(det_data, training)
        pose_data = {'inputs': data['inputs'], 'data_samples': data['pose_samples']}
        pose_processed = self.pose_preprocessor(pose_data, training)
        det_processed['pose_samples'] = pose_processed['data_samples']
        return {
            'inputs': det_processed['inputs'],
            'det_samples': det_processed['data_samples'],
            'pose_samples': pose_processed['data_samples']
        }