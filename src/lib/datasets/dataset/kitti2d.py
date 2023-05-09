from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class KITTI2D(data.Dataset):
    num_classes = 3
    default_resolution = [384, 1280]
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(KITTI2D, self).__init__()
        self.data_dir = '/store/datasets/KITTI/left_image'
        self.img_dir = self.data_dir

        if split == 'test':
            self.annot_path = '/store/datasets/KITTI/left_image/testval.json'
        elif split == 'val':
            self.annot_path = '/store/datasets/KITTI/left_image/val.json'
        else:
            self.annot_path = '/store/datasets/KITTI/left_image/train.json'

        self.max_objs = 50
        self.class_name = ['__background__', 'Pedestrian', 'Car', 'Cyclist']
        self._valid_ids = [
            1, 2, 3]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing UA-Detrac {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    # def save_results(self, results, save_dir):
    #     json.dump(self.convert_eval_format(results),
    #               open('{}/results.json'.format(save_dir), 'w'))

    def save_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for img_id in results.keys():
            out_path = os.path.join(results_dir, str(int(img_id)+5985).zfill(6) + '.txt')
            f = open(out_path, 'w')
            for cls_ind in results[img_id]:
                for j in range(len(results[img_id][cls_ind])):
                    class_name = self.class_name[cls_ind]
                    f.write('{} 0.0 0.0 0.0'.format(class_name))
                    for i in range(len(results[img_id][cls_ind][j])):
                        f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
                        if i == 3:
                            # f.write(' 0 0 0 0 0 0 0')
                            f.write(' -1 -1 -1 -1000 -1000 -1000 -10')

                    f.write('\n')
            f.close()
    # def run_eval(self, results, save_dir):
    #     # result_json = os.path.join(save_dir, "results.json")
    #     # detections  = self.convert_eval_format(results)
    #     # json.dump(detections, open(result_json, "w"))
    #     self.save_results(results, save_dir)
    #     coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    #     coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        os.system('/store/datasets/KITTI/eval/cpp/evaluate_object_offline ' + \
                  '/store/datasets/KITTI/left_image/labels_testval ' + \
                  '{}/results/'.format(save_dir))