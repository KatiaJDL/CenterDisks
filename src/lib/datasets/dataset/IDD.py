from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageChops
import torch.utils.data as data
import glob
from multiprocessing import Pool
from pycocotools.cocoeval import COCOeval

def individual_gaussian(heatmap, centers, radius, ceiling = None):

  H, W = heatmap.shape

  for k in range(0,len(centers)):

    #print(np.square([[centers[k][0]-i for i in range(W)]]).shape)
    #print(np.square([[centers[k][1]-j] for j in range(H)]).shape)
    #print(centers)
    #print(radius)

    if type(radius) == list:
        heatmap += np.exp( - (np.square([[centers[k][0]-i for i in range(W)]]) \
                + np.square([[centers[k][1]-j] for j in range(H)])) /(2*radius[k%len(radius)]**2))

    else: 

        heatmap += np.exp( - (np.square([[centers[k][0]-i for i in range(W)]]) \
                + np.square([[centers[k][1]-j] for j in range(H)])) /(2*radius**2))

    #print(np.array_equal(heatmap, np.zeros_like(heatmap)))

  #ax = sns.heatmap(heatmap)
  #plt.show()
  
  if ceiling == 'clamp':
    heatmap = torch.clip(heatmap, 0, 1)
  elif ceiling == 'tanh':
    heatmap = np.tanh(heatmap)


  return heatmap


class IDD(data.Dataset):
    num_classes = 9
    # default_resolution = [1024, 2048]
    default_resolution = [512, 1024]
    # default_resolution = [512, 512]

    mean = np.array([0.28404999637454165, 0.32266921542410754, 0.2816898182839038], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.04230349568017417, 0.04088212241688149, 0.04269893084955519],dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(IDD, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        self.split = split
        self.opt = opt

        base_dir = '../IDDStuff/BBoxes'

        if split == 'test':
            self.annot_path = os.path.join(base_dir, 'test.json')
        elif split == 'val':
            self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_regular_interval.json')
            # self.annot_path = os.path.join(base_dir, 'val' + str(self.opt.nbr_points) + '_real_points.json')
        else:
            self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_regular_interval.json')
            # self.annot_path = os.path.join(base_dir, 'train' + str(self.opt.nbr_points) + '_real_points.json')

        self.max_objs = 128
        self.class_name = [
            '__background__', 'person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback']
        self.label_to_id = {'person':6, 'rider':8, 'motorcycle':9, 'bicycle':10, 'autorickshaw':11, 'car':12, 'truck':13, 'bus':14, 'vehicle fallback':18}

        # to customize
        self.class_frequencies = {'person': 0.15, 'rider': 0.03, 'car': 0.20, 'truck': 0.03, 'bus': 0.03, 'motorcycle': 0.03, 'bicycle': 0.03, 'autorickshaw': 0.33, 'vehicle fallback': 0.18,}

        self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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



        print('==> initializing IDD {} data.'.format(split))
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

    def convert_polygon_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    score = bbox[4]
                    depth = bbox[-1]
                    label = self.class_name[cls_ind]
                    polygon = list(map(self._to_float, bbox[5:-1]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "polygon": polygon,
                        "score": float("{:.2f}".format(score)),
                        "depth": float(depth),
                    }
                    detections.append(detection)
        return detections

    def format_and_write_to_IDD(self, all_bboxes, save_dir):
        id_to_file = {}
        anno = json.load(open(self.annot_path))
        for image in anno['images']:
            id_to_file[image['id']] = image['file_name']

        for image_id in all_bboxes:
            image_name = id_to_file[int(image_id)]
            w, h = Image.open(image_name).size
            sub_dir = image_name.split('/')[-2]
            write_dir = os.path.join(save_dir, sub_dir)
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
            text_file = open(os.path.join(write_dir, os.path.basename(image_name).replace('.png', '.txt')), 'w')
            count = 0
            for cls_ind in all_bboxes[image_id]:
                param_list = []
                to_remove_mask = Image.new('L', (w, h), 1)
                for bbox in all_bboxes[image_id][cls_ind]:
                    if bbox[4] >= self.opt.thresh:
                        score = str(bbox[4])
                        depth = bbox[-1]
                        label = self.class_name[cls_ind]
                        polygon = list(map(self._to_float, bbox[5:-1]))
                        # poly_points = []
                        # for i in range(0, len(polygon)-1, 2):
                        #     poly_points.append((int(polygon[i]), int(polygon[i+1])))
                        # polygon_mask = Image.new('L', (2048, 1024), 0)
                        # ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
                        mask_path = os.path.join(
                            os.path.basename(image_name).replace('.png', '_' + str(count) + '.png'))
                        # polygon_mask.save(mask_path)
                        text_file.write(
                            os.path.basename(mask_path) + ' ' + str(self.label_to_id[label]) + ' ' + score + '\n')
                        count += 1
                        param_list.append((polygon, mask_path, bbox[4], depth))

                for args in sorted(param_list, key=lambda x: x[-1]):
                    polygon, mask_path, score, depth = args
                    poly_points = []
                    for i in range(0, len(polygon), 2):
                        poly_points.append((int(polygon[i]), int(polygon[i + 1])))
                    polygon_mask = Image.new('L', (w, h), 0)
                    ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
                    polygon_mask = Image.fromarray(np.array(polygon_mask) * np.array(to_remove_mask))
                    if score >= 0.5:
                        ImageDraw.Draw(to_remove_mask).polygon(poly_points, outline=0, fill=0)
                    polygon_mask.save(os.path.join(write_dir, mask_path))

    def format_and_write_to_IDD_gaussian(self, all_bboxes, save_dir):
        id_to_file = {}
        anno = json.load(open(self.annot_path))
        for image in anno['images']:
            id_to_file[image['id']] = image['file_name']

        for image_id in all_bboxes:
            image_name = id_to_file[int(image_id)]
            w, h = Image.open(image_name).size
            sub_dir = image_name.split('/')[-2]
            write_dir = os.path.join(save_dir, sub_dir)
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
            text_file = open(os.path.join(write_dir, os.path.basename(image_name).replace('.png', '.txt')), 'w')
            count = 0
            for cls_ind in all_bboxes[image_id]:
                param_list = []
                to_remove_mask = Image.new('L', (w, h), 1)
                for bbox in all_bboxes[image_id][cls_ind]:
                    if bbox[4] >= self.opt.thresh:
                        score = str(bbox[4])
                        depth = bbox[-1]
                        if self.opt.r_variation == 'one':
                            radius = float(bbox[-2])
                            centers = list(map(self._to_float, bbox[5:-2]))
                        else:
                            centers = list(map(self._to_float, bbox[5:5+2*self.opt.nbr_points]))
                            radius = list(map(self._to_float, bbox[5+2*self.opt.nbr_points:-1]))
                        label = self.class_name[cls_ind]
                        mask_path = os.path.join(
                            os.path.basename(image_name).replace('.png', '_' + str(count) + '.png'))
                        text_file.write(
                            os.path.basename(mask_path) + ' ' + str(self.label_to_id[label]) + ' ' + score + '\n')
                        count += 1
                        centers = [(int(x), int(y)) for x, y in zip(centers[0::2], centers[1::2])]
                        param_list.append((centers, radius, bbox[4], mask_path, depth))

                for args in sorted(param_list, key=lambda x: x[-1]):
                    centers, r, score, mask_path, depth = args

                    pred_gaussian = np.zeros((h, w))
                    pred_gaussian = individual_gaussian(pred_gaussian, centers, r, self.opt.gaussian_ceiling)

                    pred_gaussian = (pred_gaussian>self.opt.threshold)

                    gaussian_img = Image.fromarray((pred_gaussian*255).astype(np.uint8))

                    
                    if score >= 0.5:
                        to_remove_mask += np.array(gaussian_img)
                        to_remove_mask[to_remove_mask > 0] = 1
                    gaussian_img.save(os.path.join(write_dir, mask_path))

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        if self.opt.task == 'polydet':
            json.dump(self.convert_polygon_eval_format(results),
                      open('{}/results.json'.format(save_dir), 'w'))
        else:
            json.dump(self.convert_eval_format(results),
                      open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        if self.opt.task == 'ctdet':
            self.save_results(results, save_dir)
            res_dir = os.path.join(save_dir, 'results_')
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            self.format_and_write_to_IDD(results, res_dir)
            coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
            coco_eval = COCOeval(self.coco, coco_dets, "bbox")
            # coco_eval.params.catIds = [2, 3, 4, 6, 7, 8, 10, 11, 12, 13]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        elif self.opt.task == 'polydet':
            self.save_results(results, save_dir)
            res_dir = os.path.join(save_dir, 'results')
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            to_delete = os.path.join(save_dir, 'results/*.txt')
            files = glob.glob(to_delete)
            for f in files:
                os.remove(f)
            to_delete = os.path.join(save_dir, 'results/*/*.png')
            files = glob.glob(to_delete)
            for f in files:
                os.remove(f)
            self.format_and_write_to_IDD(results, res_dir)
            os.environ['IDD_DATASET'] = '/store/datasets/IDD'
            os.environ['IDD_RESULTS'] = res_dir
            from datasets.evaluation.IDDscripts.evaluation import evaluate_instance_segmentation
            AP = evaluate_instance_segmentation.getAP()
            return AP
            # return 0
        elif self.opt.task == 'gaussiandet':
            self.save_results(results, save_dir)
            res_dir = os.path.join(save_dir, 'results')
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            to_delete = os.path.join(save_dir, 'results/*.txt')
            files = glob.glob(to_delete)
            for f in files:
                os.remove(f)
            to_delete = os.path.join(save_dir, 'results/*/*.png')
            files = glob.glob(to_delete)
            for f in files:
                os.remove(f)
            #print("format and write")
            self.format_and_write_to_IDD_gaussian(results, res_dir) 
            os.environ['IDD_DATASET'] = '/store/datasets/IDD'
            os.environ['IDD_RESULTS'] = res_dir
            from datasets.evaluation.IDDscripts.evaluation import evaluate_instance_segmentation
            AP = evaluate_instance_segmentation.getAP()
            return AP
