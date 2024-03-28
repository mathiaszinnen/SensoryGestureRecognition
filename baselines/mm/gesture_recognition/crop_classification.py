from mmpretrain.apis.image_classification import ImageClassificationInferencer
from typing import Union
from math import ceil
from mmpretrain.utils import track
import numpy as np


InputType = Union[str, np.ndarray, list]


class CropClassificationInferencer(ImageClassificationInferencer):

    def __call__(self,
                 inputs: InputType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)


        preds = []
        for data in track(
                inputs, 'Inference', total=ceil(len(ori_inputs) / batch_size)):
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(ori_inputs, preds, **visualize_kwargs)
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results

        
