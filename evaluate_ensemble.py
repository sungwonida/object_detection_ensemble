# AUTHOR: David Jung
# EMAIL: sungwonida@gmail.com
# DATE: 2020-09-06

import os
import sys
import json
import cv2
import torch
import numpy as np
from typing import Dict, List
import logging
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, Instances
from config import *


class BetterStructOutput:
    """ Reconstruct the detection outputs so could make some specific jobs done faster.

        dets_dict: A dictionary on which put object detection outputs together while make them reconstructed
            Structure. {image_id: {category_id: {det_ids: {'bboxes': [...], 'scores': [...]}}}}
                Example.
                   {139:  # image_id
                        {56:  # category_id
                            {0:  # det_id
                                {'bboxes': [[294.155242, 218.248077, 58.981506, 98.897247],
                                           [293.428253, 221.404052, 58.089050, 93.993896],
                                           [292.919769, 214.956222, 59.479888, 101.260543]]
                                 'scores': [0.989548, 0.993888, 0.994455]
                                }
                            {1:
                                ...
                         72:
                            ...
                        }
                    }
        weights: Weights of each object detection models
            Example.
                [34.353, 36.667, 38.400]
    """

    def __init__(self, dets: List = None, reverse_id_mapping=None, **kwargs):
        self.reverse_id_mapping = reverse_id_mapping
        self._data = self._parse_dets(dets)

    def __call__(self, dets: List):
        self._data = self._parse_dets(dets)
        return self

    def _construct_det_id(self, bbox, score):
        return {'bboxes': [bbox], 'scores': [score]}

    def _construct_category_id(self, det_id, bbox, score):
        return {det_id: self._construct_det_id(bbox, score)}

    def _construct_image_id(self, cat_id, det_id, bbox, score):
        return {cat_id: self._construct_category_id(det_id, bbox, score)}

    def _parse_dets(self, dets):
        if dets is None:
            return

        dets_dict = {}
        for i, det in enumerate(dets):
            for d in det:
                image_id, cat_id, bbox, score = d['image_id'], d['category_id'], d['bbox'], d['score']
                if self.reverse_id_mapping:
                    index = list(self.reverse_id_mapping.values()).index(cat_id)
                    cat_id = list(self.reverse_id_mapping.keys())[index]

                if image_id not in dets_dict.keys():
                    dets_dict[image_id] = self._construct_image_id(
                        cat_id, i, bbox, score)
                else:
                    if cat_id not in dets_dict[image_id].keys():
                        dets_dict[image_id][cat_id] = self._construct_category_id(
                            i, bbox, score)
                    else:
                        if i not in dets_dict[image_id][cat_id].keys():
                            dets_dict[image_id][cat_id][i] = self._construct_det_id(
                                bbox, score)
                        else:
                            dets_dict[image_id][cat_id][i]['bboxes'].append(
                                bbox)
                            dets_dict[image_id][cat_id][i]['scores'].append(
                                score)

        self._validate(dets_dict)
        return dets_dict

    def _validate(self, d):
        for image_data in d.values():
            for cat_data in image_data.values():
                for det_data in cat_data.values():
                    assert (len(det_data['bboxes']) == len(det_data['scores']))

    def data(self):
        return self._data


class COCOStruct(BetterStructOutput):
    reverse_id_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
                          40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

    def __init__(self, dets: List = None, **kwargs):
        super().__init__(dets, reverse_id_mapping=self.reverse_id_mapping, kwargs=kwargs)


class COCODetectionEnsemble:
    """ Make an ensemble from outputs of object detection models.

        Gets its source from instance of BetterStructOutput.
    """

    def __init__(self,
                 score_thresh: float = 0.0,
                 iou_thresh: float = 0.5,
                 weights: List = None):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.weights = np.array(weights) if weights else None

    def __call__(self, dets: Dict) -> Dict:
        return self.ensemble(dets)

    def ensemble_category(self, dets_category: Dict) -> Dict:
        ndets = len(dets_category.keys())
        out = {'pred_boxes': [], 'scores': []}
        used = set()

        if self.weights is None:
            w = 1 / float(ndets)
            weights = [w] * ndets
        else:
            weights = self.weights[list(dets_category.keys())]
            assert(len(weights) == ndets)
            weights /= sum(weights)

        weights = dict(zip(dets_category.keys(), weights))

        for id1, det1 in dets_category.items():
            bboxes = det1['bboxes']
            scores = det1['scores']

            for i in range(len(bboxes)):
                box = bboxes[i]
                score = scores[i]
                query = str(id1) + str(box) + str(score)
                # print(f"query: {query} ({query in used})")
                if query in used:
                    continue

                used.add(query)
                found = []

                for id2, det2 in dets_category.items():
                    if id1 == id2:
                        continue

                    bestiou = self.iou_thresh
                    bestbox = None
                    bestscore = None

                    obboxes = det2['bboxes']
                    oscores = det2['scores']

                    for j in range(len(obboxes)):
                        obox = obboxes[j]
                        oscore = oscores[j]
                        oquery = str(id2) + str(obox) + str(oscore)
                        # print(f"oquery: {oquery} ({oquery in used})")
                        if oquery in used:
                            continue

                        iou = calculate_iou(box, obox)
                        if iou > bestiou:
                            bestiou = iou
                            bestbox = obox
                            bestscore = oscore

                    if bestbox is not None:
                        # print(f"weights: {weights}, id2: {id2}, keys: {dets_category.keys()}")
                        found.append((bestbox, bestscore, weights[id2]))
                        bestquery = str(id2) + str(bestbox) + str(bestscore)
                        used.add(bestquery)

                if len(found) > 0:
                    allboxes = [(box, score, weights[id1])]
                    allboxes.extend(found)

                    bx = 0.0
                    by = 0.0
                    bw = 0.0
                    bh = 0.0
                    score = 0.0
                    wsum = 0.0

                    for bb in allboxes:
                        # weight
                        w = bb[2]
                        wsum += w

                        # (box_x, box_y, box_w, box_h)
                        b = bb[0]
                        bx += w*b[0]
                        by += w*b[1]
                        bw += w*b[2]
                        bh += w*b[3]

                        # score
                        score += w*bb[1]

                    bx /= wsum
                    by /= wsum
                    bw /= wsum
                    bh /= wsum

                    if score >= self.score_thresh:
                        out['pred_boxes'].append([bx, by, bw, bh])
                        out['scores'].append(score)

                else:
                    score /= ndets
                    if score >= self.score_thresh:
                        out['pred_boxes'].append(list(box))
                        out['scores'].append(score)

        return out

    def ensemble_image(self, dets_image: Dict) -> Dict:
        out = {'pred_boxes': [], 'scores': [], 'pred_classes': []}

        for cat in dets_image.keys():
            ret = self.ensemble_category(dets_image[cat])

            # convert
            # from: [dim1, dim2, len1, len2]
            # to  : [dim1, dim2, dim1+len1, dim2+len2]
            bboxes = np.array(ret['pred_boxes'])
            if len(bboxes) > 0:
                bboxes[:, 2:] += bboxes[:, :2]
            out['pred_boxes'] += bboxes.tolist()
            out['scores'] += ret['scores']
            out['pred_classes'] += [int(cat)] * len(ret['pred_boxes'])

        return out

    def ensemble(self, dets: Dict) -> Dict:
        out = {}

        for image_id in dets.keys():
            ret = self.ensemble_image(dets[image_id])
            out[image_id] = ret

        return out


class Ensemble:
    def __init__(self, *args, **kwargs):
        # delegates
        self.data_encoder = COCOStruct()
        self.ensemble_runner = COCODetectionEnsemble(*args, **kwargs)

    def ensemble(self, dets):
        if self.data_encoder:
            dets = self.data_encoder(dets).data()

        return self.ensemble_runner(dets)

    @classmethod
    def save(cls, ensemble, path):
        with open(path, 'w') as f:
            json.dump(ensemble, f)

    @classmethod
    def load(cls, path, as_instances=False, dataset_path=None, data_ext='jpg'):
        with open(path, 'rb') as f:
            ensemble_dict = json.load(f)

        if as_instances:
            ensemble = {}
            for image_id, data in ensemble_dict.items():
                filename = os.path.join(dataset_path, '.'.join(
                    [str(image_id).zfill(12), data_ext]))
                try:
                    height, width = cv2.imread(filename).shape[:2]
                except Exception as e:
                    print(f"filename: {filename}")
                    raise e
                instances = Instances((height, width))
                instances.pred_boxes = Boxes(torch.Tensor(data['pred_boxes']))
                instances.scores = torch.Tensor(data['scores'])
                instances.pred_classes = torch.LongTensor(data['pred_classes'])
                ensemble[image_id] = [{'instances': instances}]
        else:
            ensemble = ensemble_dict

        return ensemble


def calculate_iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2

    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max(
        [y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0

    intersect = (xi2-xi1) * (yi2-yi1)
    union = area1 + area2 - intersect
    return intersect / union


def evaluate_ensemble(infer_output, data_loader, evaluator):
    """
    Evaluate precomputed output using data_loader and evaluate the metrics with evaluator.

    Args:
        infer_output: a dictionary that holds image_id as key and Instances object as its value
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    evaluator.reset()

    for idx, inputs in enumerate(data_loader):
        filename = inputs[0]['file_name']
        basename = os.path.basename(filename)
        image_id = str(int(os.path.splitext(basename)[0]))

        if image_id in infer_output.keys():
            outputs = infer_output[image_id]
        else:
            height, width = cv2.imread(filename).shape[:2]
            instances = Instances((height, width))
            instances.pred_boxes = Boxes(torch.Tensor([]))
            instances.scores = torch.Tensor([])
            instances.pred_classes = torch.LongTensor([])
            outputs = [{'instances': instances}]
            # print(f"empty: {image_id}")

        evaluator.process(inputs, outputs)

        if idx % 10 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

    print()
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


if __name__ == '__main__':
    dets = []

    for path in model_eval_paths:
        with open(path, 'rb') as f:
            print(f"Loaded {path}")
            dets.append(json.load(f))

    output = Ensemble(iou_thresh=0.1, weights=weights).ensemble(dets)
    save_path = 'output/ensemble_results.json'
    Ensemble.save(output, save_path)
    print(f"Saved the ensemble result to {save_path}")

    print("Preparing for evaluating the ensemble")
    cfg = get_cfg()
    # cfg.MODEL.DEVICE = "cpu"
    dataset_path = os.path.join('datasets', 'coco', 'val2017')
    det_output = Ensemble.load('output/ensemble_results.json',
                               as_instances=True,
                               dataset_path=dataset_path)
    evaluator = COCOEvaluator("coco_2017_val", cfg,
                              False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")
    print(evaluate_ensemble(det_output, val_loader, evaluator))
