# Anchor generation
# Tensor with multi-dimensional ground truth here  for the RPN stage that contains a flag
# indicating whether the anchor should be included in training, whether it is
# an object, and the box delta regression targets.

import itertools
import numpy as np
from math import sqrt
from . import math_utils


def _compute_anchor_sizes():
  #
  # Anchor scales and aspect ratios.
  #
  # x * y = area          x * (x_aspect * x) = x_aspect * x^2 = area
  # x_aspect * x = y  ->  x = sqrt(area / x_aspect)
  #                       y = x_aspect * sqrt(area / x_aspect)
  #
  areas = [ 128*128, 256*256, 512*512 ]   # pixels
  x_aspects = [ 0.5, 1.0, 2.0 ]           # x:1 ratio

  # Generate all 9 combinations of area and aspect ratio
  heights = np.array([ x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])
  widths = np.array([ sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])

  # Return as (9,2) matrix of sizes
  return np.vstack([ heights, widths ]).T

def generate_anchor_maps(image_shape, feature_map_shape, feature_pixels):
  # Generates maps defining the anchors for a given input image size. There are 9
  # different anchors at each feature map cell (3 scales, 3 ratios).

  # Parameters
  # image_shape : Tuple[int, int, int]
  #   Shape of the input image, (channels, height, width), at the scale it will
  #   be passed into the Faster R-CNN model.
  # feature_map_shape : Tuple[int, int, int]
  #   Shape of the output feature map, (channels, height, width).
  # feature_pixels : int
  #   Distance in pixels between anchors. This is the size, in input image space,
  #   of each cell of the feature map output by the feature extractor stage of
  #   the Faster R-CNN network.

  # Returns
  # np.ndarray (center_y, center_x, anchor_height, anchor_width),
  # np.ndarray (height, width, num_anchors) 

  assert len(image_shape) == 3

  # Base anchor template: (num_anchors,4), with each anchor being specified by
  # its corners (y1,x1,y2,x2)
  anchor_sizes = _compute_anchor_sizes()
  num_anchors = anchor_sizes.shape[0]
  anchor_template = np.empty((num_anchors, 4))
  anchor_template[:,0:2] = -0.5 * anchor_sizes  # y1, x1 (top-left)
  anchor_template[:,2:4] = +0.5 * anchor_sizes  # y2, x2 (bottom-right)

  # Shape of map, (H,W), determined by feature extractor backbone
  height = feature_map_shape[-2]  # index from back in case batch dimension is supplied
  width = feature_map_shape[-1]

  # Generate (H,W,2) map of coordinates, in feature space, each being [y,x]
  y_cell_coords = np.arange(height)
  x_cell_coords = np.arange(width)
  cell_coords = np.array(np.meshgrid(y_cell_coords, x_cell_coords)).transpose([2, 1, 0])

  # Convert all coordinates to image space (pixels) at *center* of each cell
  center_points = cell_coords * feature_pixels + 0.5 * feature_pixels

  # (H,W,2) -> (H,W,4), repeating the last dimension so it contains (y,x,y,x)
  center_points = np.tile(center_points, reps = 2)

  # (H,W,4) -> (H,W,4*num_anchors)
  center_points = np.tile(center_points, reps = num_anchors)

  #
  # Now we can create the anchors by adding the anchor template to each cell
  # location. Anchor template is flattened to size num_anchors * 4 to make
  # the addition possible (along the last dimension).
  #
  anchors = center_points.astype(np.float32) + anchor_template.flatten()

  # (H,W,4*num_anchors) -> (H*W*num_anchors,4)
  anchors = anchors.reshape((height*width*num_anchors, 4))

  # Valid anchors are those that do not cross image boundaries
  image_height, image_width = image_shape[1:]
  valid = np.all((anchors[:,0:2] >= [0,0]) & (anchors[:,2:4] <= [image_height,image_width]), axis = 1)

  # Convert anchors to anchor format: (center_y, center_x, height, width)
  anchor_map = np.empty((anchors.shape[0], 4))
  anchor_map[:,0:2] = 0.5 * (anchors[:,0:2] + anchors[:,2:4])
  anchor_map[:,2:4] = anchors[:,2:4] - anchors[:,0:2]

  # Reshape maps and return
  anchor_map = anchor_map.reshape((height, width, num_anchors * 4))
  anchor_valid_map = valid.reshape((height, width, num_anchors))
  return anchor_map.astype(np.float32), anchor_valid_map.astype(np.float32)

def generate_rpn_map(anchor_map, anchor_valid_map, gt_boxes, object_iou_threshold = 0.7, background_iou_threshold = 0.3):

  # Generates a map containing ground truth data for training the region proposal network.

  height, width, num_anchors = anchor_valid_map.shape

  # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
  gt_box_corners = np.array([ box.corners for box in gt_boxes ])
  num_gt_boxes = len(gt_boxes)

  # Compute ground truth box center points and side lengths
  gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])
  gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]

  # Flatten anchor boxes to (N,4) and convert to corners
  anchor_map = anchor_map.reshape((-1,4))
  anchors = np.empty(anchor_map.shape)
  anchors[:,0:2] = anchor_map[:,0:2] - 0.5 * anchor_map[:,2:4]  # y1, x1
  anchors[:,2:4] = anchor_map[:,0:2] + 0.5 * anchor_map[:,2:4]  # y2, x2
  n = anchors.shape[0]

  # Initialize all anchors initially as negative (background). 
  objectness_score = np.full(n, -1)   
  gt_box_assignments = np.full(n, -1) # -1 means no box

  # Compute IoU between each anchor and each ground truth box, (N,M).
  ious = math_utils.intersection_over_union(boxes1 = anchors, boxes2 = gt_box_corners)

  # Need to remove anchors that are invalid (straddle image boundaries) from
  # consideration entirely and the easiest way to do this is to wipe out their
  # IoU scores
  ious[anchor_valid_map.flatten() == 0, :] = -1.0

  # Find the best IoU ground truth box for each anchor and the best IoU anchor
  # for each ground truth box.

  max_iou_per_anchor = np.max(ious, axis = 1)           
  best_box_idx_per_anchor = np.argmax(ious, axis = 1)   
  max_iou_per_gt_box = np.max(ious, axis = 0)          
  highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0] 

  # Anchors below the minimum threshold are negative
  # meet the threshold IoU are positive
  # overlap the most with ground truth boxes are positive
  objectness_score[max_iou_per_anchor < background_iou_threshold] = 0
  objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1
  objectness_score[highest_iou_anchor_idxs] = 1


  # Assign the highest IoU ground truth box to each anchor. Anchors that are to be ignored will be marked invalid.
  gt_box_assignments[:] = best_box_idx_per_anchor
  enable_mask = (objectness_score >= 0).astype(np.float32)
  objectness_score[objectness_score < 0] = 0


  # Compute box delta regression targets for each anchor
  box_delta_targets = np.empty((n, 4))
  box_delta_targets[:,0:2] = (gt_box_centers[gt_box_assignments] - anchor_map[:,0:2]) / anchor_map[:,2:4] 
  box_delta_targets[:,2:4] = np.log(gt_box_sides[gt_box_assignments] / anchor_map[:,2:4])                 

  # RPN gt map
  rpn_map = np.zeros((height, width, num_anchors, 6))
  rpn_map[:,:,:,0] = anchor_valid_map * enable_mask.reshape((height,width,num_anchors))  
  rpn_map[:,:,:,1] = objectness_score.reshape((height,width,num_anchors))
  rpn_map[:,:,:,2:6] = box_delta_targets.reshape((height,width,num_anchors,4))

  # Return map along with positive and negative anchors
  rpn_map_coords = np.transpose(np.mgrid[0:height,0:width,0:num_anchors], (1,2,3,0))                  
  object_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] > 0) & (rpn_map[:,:,:,0] > 0))]      
  background_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] == 0) & (rpn_map[:,:,:,0] > 0))] 

  return rpn_map.astype(np.float32), object_anchor_idxs, background_anchor_idxs
