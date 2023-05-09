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

base_dir = '/store/datasets/cityscapes'
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

results_file = '/store/travail/kajoda/results.json'
results = json.load(open(results_file, 'r'))
image_to_boxes = {}
num_radius = 0
for result in results:
    box = [result['score']]
    box += [result['category_id']]
    if 'disks' in result:
        box += [result['depth']]
        box += list(result['disks'])
        if type(result['radius']) == list:
            num_radius = len(list(result['radius']))
            box += list(result['radius'])
        else:
            num_radius = 1
            box += [result['radius']]
    else:
        # box += [result['category_id']]
        x0, y0, w, h = list(result['bbox'])
        x1, y1 = x0 + w, y0 + h
        box += [x0, y0, x1, y0, x1, y1, x0, y1]

    if id_to_file[result['image_id']] in image_to_boxes:
        image_to_boxes[id_to_file[result['image_id']]].append(box)
    else:
        image_to_boxes[id_to_file[result['image_id']]] = [box]

# fig, ax = plt.subplots(1)
# set_size_inches(10, 6)

count = 1
for key in sorted(image_to_boxes):

    # im = np.array(Image.open(os.path.join(base_dir, key)), dtype=np.uint8)
    im = Image.open(os.path.join(base_dir, key))
    depth_map = Image.fromarray(np.ones((im.size[1], im.size[0])) * 255)
    depths = []
    # ax.imshow(im)
    # Hide grid lines
    # ax.grid(False)

    # Hide axes ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    for poly in sorted(image_to_boxes[key], key = lambda x: x[2], reverse=True):
        score = float(poly[0])
        if score >= TRESH:
            depth = float(poly[2])
            depths.append(depth)
            label = int(poly[1]) - 1
            ec = (255, 255, 0, 100)
            if label == 0:
                ec = (255, 255, 0, 100)  # person
            elif label == 1:
                ec = (255, 127, 0, 100)  # rider
            elif label == 2:
                ec = (0, 149, 255, 100)  # car
            elif label == 3:
                ec =(107, 35, 143, 100)  # truck
            elif label == 4:
                ec = (255, 0, 0, 100)  # bus
            elif label == 5:
                ec = (170, 0, 255, 100)  # train
            elif label == 6:
                ec = (255, 0, 170, 100)  # motorcycle
            elif label == 7:
                ec = (220, 185, 237, 100)  # bicycle
            elif label == 8:
                ec = (0, 0, 0, 100)  # pole
            elif label == 9:
                ec = (0, 0, 0, 100)  # traffic sign
            points = []
            for i in range(3, len(poly)-1- num_radius, 2):
                points.append((poly[i], poly[i+1]))
            # x, y = points[0::2], points[1::2]
            # try:
            #     polygon = Polygon((points))
            #     polygon = polygon.buffer(10, join_style=1).buffer(-10.0, join_style=1)
            #     x, y = polygon.exterior.coords.xy
            #     points = [(int(item[0]), int(item[1])) for item in zip(x, y)]
            # except:
            #     do_nothing = True

            for k, point in enumerate(points):
                x, y = point[0], point[1]
                r = 1.2*math.sqrt(2*math.log(2)*poly[len(poly)-num_radius:][k % num_radius]**2)
                ImageDraw.Draw(im, 'RGBA').ellipse([(x-r, y-r),(x+r, y+r)], outline=0, fill=ec)
            # contour = list(bresenham.bresenham(points[-1][0], points[-1][1], points[0][0], points[0][1]))
            # for i in range(len(points)-1):
            #     line = bresenham.bresenham(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
            #     contour += line
            # for point in set(contour):
            #     ImageDraw.Draw(im, 'RGBA').ellipse([(point[0]-5, point[1]-5), (point[0]+5, point[1]+5)], outline=0, fill=ec)

            # ImageDraw.Draw(im, 'RGBA').line(points[0:2], fill=(0, 0, 0), width=2)
    for poly in sorted(image_to_boxes[key], key=lambda x: x[2], reverse=True):
        score = float(poly[0])
        depth = float(poly[2])
        if score >= TRESH:
            points = []
            for i in range(3, len(poly) - 1, 2):
                points.append((poly[i], poly[i + 1]))
            depth_color = (((depth-np.min(depths)) / np.max(depths))) * 255
            for k, point in enumerate(points):
                x, y = point[0], point[1]
                r = 1.2*math.sqrt(2*math.log(2)*poly[len(poly)-num_radius:][k % num_radius]**2)
                ImageDraw.Draw(depth_map).ellipse([(x-r, y-r),(x+r, y+r)], outline=0, fill=depth_color)

            # poly = patches.Polygon(points, linewidth=lw, edgecolor=ec, facecolor='none')
            # ax.add_patch(poly)
    # im.show()
    if not os.path.exists(os.path.join(os.path.dirname(results_file), 'image_examples')):
        os.mkdir(os.path.join(os.path.dirname(results_file), 'image_examples'))
    im.save(os.path.join(os.path.dirname(results_file), 'image_examples', os.path.basename(key)))
    heatmap = cv2.applyColorMap(np.array(depth_map).astype(np.uint8), cv2.COLORMAP_HOT)
    cv2.imwrite(os.path.join(os.path.dirname(results_file), 'image_examples', os.path.basename(key).replace('.png', '_depth.png')), heatmap)
    # plt.show(block=False)
    # plt.savefig(os.path.join(os.path.dirname(results_file), 'image_examples', os.path.basename(key)))
    # plt.pause(0.0001)
    # ax.cla()
