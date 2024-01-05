# PASCAL VOC dataset loader. Datasets available at: http://host.robots.ox.ac.uk/pascal/VOC/
#
# The dataset directory must contain the following sub-directories:
#
# project_directory/
# │
# ├── Annotations/
# │   ├── file1.xml
# │
# │
# ├── ImageSets/
# │   ├── Layout/
# │   │   ├── train.txt(contains all the name of files without ".xml")
# │   │   └── test.txt
# │   └── Main/
# │       ├── class_train.txt
# │       └── class_test.txt
# │
# └── JPEGImages/
#     ├── image1.jpg
#
# Remember to change the dir to your dir of VOC dataset in _main_ or in the bash script.

import os
import random
import numpy as np
import xml.etree.ElementTree as xET

from pathlib import Path
from typing import List
from typing import Tuple
from dataclasses import dataclass

from . import image
from .training_data import Box
from .training_data import Training
from pytorch.FasterRCNN.models import anchors

# basic information of the dataset
class Dataset:  
  num_classes = 21
  class_index_to_name = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
  }

  def __iter__(self):
    self._i = 0
    if self._shuffle:
      random.shuffle(self._iterable_filepaths)
    return self
  
  def __init__(self, split, image_preprocessing_params, compute_feature_map_shape_fn, 
feature_pixels = 16, dir = "/home/stu5/ranshui/FasterRCNN/VOC", augment = True, shuffle = True, 
allow_difficult = False, cache = True):

    # compute_feature_map_shape_fn : Callable[Tuple[int, int, int], Tuple[int, int, int]]
    #   compute feature map shape from input (channels, height, width).
    # feature_pixels : Size of each cell in the Faster R-CNN feature map, also the separation distance between anchors.
    # augment : horizontally flip images during iteration randomly with 50% probability.
    # shuffle : whether to shuffle the dataset each time it is iterated.
    # allow_difficult : Whether to include ground truth boxes that are marked as "difficult".
    # cache : Whether to store training samples in memory after first being generated.
    
    if not os.path.exists(dir):
      raise FileNotFoundError("!Directory of dataset does not exist: %s!" % dir)
    if split not in ["train", "test"]:
      raise ValueError("!Invalid split '%s'!" % split)
    
    self.split = split
    self._dir = dir
    self.class_index_to_name = self._get_classes()

    #create a temporary dict of class name to index to be used in finding the file path
    self.class_name_to_index = {class_name: class_index for class_index, class_name in self.class_index_to_name.items()}

    self.num_classes = len(self.class_index_to_name)
    if self.num_classes != Dataset.num_classes:
      raise ValueError("!No expected number of classes (%d found but %d expected)!" % (self.num_classes, Dataset.num_classes))
    if self.class_index_to_name != Dataset.class_index_to_name:
      raise ValueError("!No expected class mapping!")

    self._filepaths = self._get_filepaths()
    self.num_samples = len(self._filepaths)
    self._gt_boxes_by_filepath = self._get_ground_truth_boxes(filepaths = self._filepaths, allow_difficult = allow_difficult)
    self._i = 0
    self._iterable_filepaths = self._filepaths.copy()
    self._image_preprocessing_params = image_preprocessing_params
    self._compute_feature_map_shape_fn = compute_feature_map_shape_fn
    self._feature_pixels = feature_pixels
    self._augment = augment
    self._shuffle = shuffle
    self._cache = cache
    self._unaugmented_cached_sample_by_filepath = {}
    self._augmented_cached_sample_by_filepath = {}



  def __next__(self):
    if self._i >= len(self._iterable_filepaths):
      raise StopIteration

    # Next file to load
    filepath = self._iterable_filepaths[self._i]
    self._i = 1 + self._i

    # Augment or not
    if self._augment:
      flip = random.randint(0, 1) != 0
    else:
      flip = False
    if flip:
      cached_sample_by_filepath = self._augmented_cached_sample_by_filepath
    else:
      cached_sample_by_filepath = self._unaugmented_cached_sample_by_filepath


    # If caching, write back to cache
    if filepath in cached_sample_by_filepath:
      sample = cached_sample_by_filepath[filepath]
    else:
      sample = self._generate_training_sample(filepath = filepath, flip = flip)

    if self._cache:
      cached_sample_by_filepath[filepath] = sample

    return sample


  def _generate_training_sample(self, filepath, flip):
    # Load and preprocess the image
    scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = filepath, preprocessing = self._image_preprocessing_params, min_dimension_pixels = 600, horizontal_flip = flip)
    _, original_height, original_width = original_shape

    # Scale boxes to new image size considering horizontal flip
    scaled_gt_boxes = []
    for box in self._gt_boxes_by_filepath[filepath]:
      if flip:
        corners = np.array([box.corners[0], original_width - box.corners[3] - 1, box.corners[2], original_width - box.corners[1] - 1])
      else:
        corners = box.corners

      scaled_box = Box(class_index = box.class_index, class_name = box.class_name, corners = corners * scale_factor)
      scaled_gt_boxes.append(scaled_box)


    # Generate anchor maps and RPN map
    anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = scaled_image_data.shape, feature_map_shape = self._compute_feature_map_shape_fn(scaled_image_data.shape), feature_pixels = self._feature_pixels)
    
    gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map = anchor_map, anchor_valid_map = anchor_valid_map, gt_boxes = scaled_gt_boxes)


    return Training(
      filepath = filepath,
      image = scaled_image,
      image_data = scaled_image_data,
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      gt_rpn_map = gt_rpn_map,
      gt_rpn_object_indices = gt_rpn_object_indices,
      gt_rpn_background_indices = gt_rpn_background_indices,
      gt_boxes = scaled_gt_boxes,
    )

  
  # get !existing! classes in the dataset
  def _get_classes(self):
    imageset_dir = os.path.join(self._dir, "ImageSets", "Main")
    classes = set([os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_" + self.split + ".txt")])

    if len(classes) == 0:
      raise ValueError("No classes found in ImageSets/Main for '%s' split" % self.split)

    # discard "background" class if present
    classes.discard("background")
    # index to class name
    class_index_to_name = {(1 + v[0]): v[1] for v in enumerate(sorted(classes))}
    # obviously value of 0 is reserved for "background"
    class_index_to_name[0] = "background"

    return class_index_to_name
  
  
  def _get_filepaths(self):
    image_list_file = os.path.join(self._dir, "ImageSets", "Layout", self.split + ".txt")
    with open(image_list_file) as fp:
      basenames = [line.strip() for line in fp.readlines()]
    image_paths = [os.path.join(self._dir, "JPEGImages", os.path.splitext(basename)[0] + ".jpg") for basename in basenames]
    return image_paths

  
  def _get_ground_truth_boxes(self, filepaths, allow_difficult):
    gt_boxes_by_filepath = {}

    for filepath in filepaths:
      num_name = os.path.splitext(os.path.basename(filepath))[0]
      annotation_file = os.path.join(self._dir, "Annotations", num_name) + ".xml"

      tree = xET.parse(annotation_file)
      root = tree.getroot()
      if tree is None:
        raise ValueError("Failed to parse %s" % annotation_file)
    

      if len(root.findall("size")) != 1:
        raise ValueError("Invalid 'size' element count in %s" % annotation_file)
      size = root.find("size")
      if len(size.findall("depth")) != 1:
        raise ValueError("Invalid 'depth' element count in %s" % annotation_file)
    

      depth = int(size.find("depth").text)
      if depth != 3:
        raise ValueError("Depth is not equal to 3 in %s" % annotation_file)

      boxes = []
      for obj in root.findall("object"):
        assert len(obj.findall("name")) == 1
        assert len(obj.findall("bndbox")) == 1
        assert len(obj.findall("difficult")) == 1

        is_difficult = int(obj.find("difficult").text) != 0
        if is_difficult and not allow_difficult:
          continue  # ignore difficult examples unless asked to include them

        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")       
        assert (
          len(bndbox.findall("xmin")) == 1
          and len(bndbox.findall("ymin")) == 1
          and len(bndbox.findall("xmax")) == 1
          and len(bndbox.findall("ymax")) == 1
        ), "Invalid bounding box format"

        x_min = int(bndbox.find("xmin").text) - 1
        y_min = int(bndbox.find("ymin").text) - 1
        x_max = int(bndbox.find("xmax").text) - 1
        y_max = int(bndbox.find("ymax").text) - 1


        corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
        box = Box(class_index = self.class_name_to_index[class_name], class_name = class_name, corners = corners)
        boxes.append(box)
        if len(boxes) <= 0:
          raise ValueError("No boxes found in %s" % annotation_file)

      gt_boxes_by_filepath[filepath] = boxes

    return gt_boxes_by_filepath
