import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import bresenham
import shapely
from shapely.geometry import Polygon
import math

TRESH = 0.5

base_dir = '/store/datasets/cityscapes/leftImg8bit/val/'
# anno = json.load(open('/store/datasets/UA-Detrac/COCO-format/test-1-on-200_b.json', 'r'))
# anno = json.load(open('../BBoxes/val16_regular_interval.json', 'r'))
# anno = json.load(open('../BBoxes/test.json', 'r'))
# anno = json.load(open('../../cityscapesStuff/BBoxes/test.json', 'r'))
# anno = json.load(open('../../cityscapesStuff/BBoxes/val16_regular_interval.json', 'r'))
anno = json.load(open('../../KITTIPolyStuff/BBoxes/test.json', 'r'))
# anno = json.load(open('../../IDDStuff/BBoxes/test.json', 'r'))
id_to_file = {}
for image in anno['images']:
    id_to_file[image['id']] = image['file_name']

results_dir = '/store/travail/kajoda/results_kitti_submit'
for filename in os.listdir(results_dir):
    complete_filename = os.path.join(results_dir, filename)
    if os.path.isfile(complete_filename):
        with open(complete_filename, 'r') as f:
            lines = f.readlines()
        adress = lines[0].split()[0].split('/')[1].split('_') #adress of mask
        filename = adress[0] + '/' + '_'.join(adress[:-1]) + '.png'
        im = Image.open(os.path.join(base_dir, filename)).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255,255,255,0))
        drawing = ImageDraw.Draw(overlay)

        lines.reverse()

        for line in lines:
            result = line.split()

            score = float(result[2])
            if score >= TRESH:
                label = int(result[1])
                ec = (255, 255, 0, 100)
                if label == 24:
                    ec = (255, 255, 0, 100)  # person
                elif label == 25:
                    ec = (255, 127, 0, 100)  # rider
                elif label == 26:
                    ec = (0, 149, 255, 100)  # car
                elif label == 27:
                    ec =(107, 35, 143, 100)  # truck
                elif label == 28:
                    ec = (255, 0, 0, 100)  # bus
                elif label == 31:
                    ec = (170, 0, 255, 100)  # train
                elif label == 32:
                    ec = (255, 0, 170, 100)  # motorcycle
                elif label == 33:
                    ec = (220, 185, 237, 100)  # bicycle
                elif label == -1:
                    ec = (0, 0, 0, 100)  # pole
                elif label == -1:
                    ec = (0, 0, 0, 100)  # traffic sign

                mask = Image.open(os.path.join(results_dir, result[0]))
                drawing.bitmap((0,0), mask, fill = ec)

        im = Image.alpha_composite(im, overlay)

        if not os.path.exists(os.path.join(results_dir, 'image_examples')):
            os.mkdir(os.path.join(results_dir, 'image_examples'))
        im.save(os.path.join(results_dir, 'image_examples', os.path.basename(filename)))
    

