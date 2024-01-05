# PyTorch implementation of the RPN (region proposal network) stage of Faster R-CNN. 

import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from . import math_utils


class RegionProposalNetwork(nn.Module):
  def __init__(self, feature_map_channels, allow_edge_proposals = False):
    super().__init__()

    # Constants
    self._allow_edge_proposals = allow_edge_proposals

    # Layers
    num_anchors = 9
    channels = feature_map_channels
    self._rpn_conv1 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = (3, 3), stride = 1, padding = "same")
    self._rpn_class = nn.Conv2d(in_channels = channels, out_channels = num_anchors, kernel_size = (1, 1), stride = 1, padding = "same")
    self._rpn_boxes = nn.Conv2d(in_channels = channels, out_channels = num_anchors * 4, kernel_size = (1, 1), stride = 1, padding = "same")

    # Initialize weights
    self._rpn_conv1.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_conv1.bias.data.zero_()
    self._rpn_class.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_class.bias.data.zero_()
    self._rpn_boxes.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_boxes.bias.data.zero_()

  def forward(self, feature_map, image_shape, anchor_map, anchor_valid_map, max_proposals_pre_nms, max_proposals_post_nms):

    # Predict objectness scores and regress region-of-interest box proposals on
    # an input feature map.

    # Parameters
    # feature_map : (batch_size, feature_map_channels, height, width).
    # image_shape : (num_channels, height, width).
    # anchor_map : (height, width, num_anchors * 4). 
    #   (center_y, center_x, height, width)
    # anchor_valid_map : (height, width, num_anchors).
    # max_proposals_pre_nms : (sorted by objectness score) 
    # max_proposals_post_nms :(sorted by objectness score) 

    # Return
    # torch.Tensor, (batch_size, height, width, num_anchors)
    # torch.Tensor, (batch_size, height, width, num_anchors * 4)
    # torch.Tensor, (y1, x1, y2, x2) box proposals, (N, 4)


    # Pass through the network
    y = F.relu(self._rpn_conv1(feature_map))
    objectness_score_map = t.sigmoid(self._rpn_class(y))
    box_deltas_map = self._rpn_boxes(y)

    # Transpose shapes to be more convenient:
    #   objectness_score_map -> (batch_size, height, width, num_anchors)
    #   box_deltas_map       -> (batch_size, height, width, num_anchors * 4)
    objectness_score_map = objectness_score_map.permute(0, 2, 3, 1).contiguous()
    box_deltas_map = box_deltas_map.permute(0, 2, 3, 1).contiguous()
    anchors, objectness_scores, box_deltas = self._extract_valid(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      objectness_score_map = objectness_score_map,
      box_deltas_map = box_deltas_map
    )

    box_deltas = box_deltas.detach()

    # Convert regressions to box corners
    proposals = math_utils.t_convert_deltas_to_boxes(
      box_deltas = box_deltas,
      anchors = t.from_numpy(anchors).cuda(),
      box_delta_means = t.tensor([0, 0, 0, 0], dtype = t.float32, device = "cuda"),
      box_delta_stds = t.tensor([1, 1, 1, 1], dtype = t.float32, device = "cuda")
    )

    sorted_indices = t.argsort(objectness_scores)                   # sort in ascending order of objectness score
    sorted_indices = sorted_indices.flip(dims = (0,))               # descending order of score
    proposals = proposals[sorted_indices][0:max_proposals_pre_nms]  # grab the top-N best proposals
    objectness_scores = objectness_scores[sorted_indices][0:max_proposals_pre_nms]  # corresponding scores

    # Clip to image boundaries
    proposals[:,0:2] = t.clamp(proposals[:,0:2], min = 0)
    proposals[:,2] = t.clamp(proposals[:,2], max = image_shape[1])
    proposals[:,3] = t.clamp(proposals[:,3], max = image_shape[2])

    # Remove anything less than 16 pixels on a side
    height = proposals[:,2] - proposals[:,0]
    width = proposals[:,3] - proposals[:,1]
    idxs = t.where((height >= 16) & (width >= 16))[0]
    proposals = proposals[idxs]
    objectness_scores = objectness_scores[idxs]

    # Perform NMS
    idxs = nms(
      boxes = proposals,
      scores = objectness_scores,
      iou_threshold = 0.7
    )
    idxs = idxs[0:max_proposals_post_nms]
    proposals = proposals[idxs]

    # Return network outputs as PyTorch tensors and extracted object proposals
    return objectness_score_map, box_deltas_map, proposals



  def _extract_valid(self, anchor_map, anchor_valid_map, objectness_score_map, box_deltas_map):
    assert objectness_score_map.shape[0] == 1 # only batch size of 1 supported for now

    height, width, num_anchors = anchor_valid_map.shape
    anchors = anchor_map.reshape((height * width * num_anchors, 4))             
    anchors_valid = anchor_valid_map.reshape((height * width * num_anchors))    
    scores = objectness_score_map.reshape((height * width * num_anchors))       
    box_deltas = box_deltas_map.reshape((height * width * num_anchors, 4))      

    if self._allow_edge_proposals:
      # Use all proposals
      return anchors, scores, box_deltas
    else:
      # Filter out those proposals generated at invalid anchors
      idxs = anchors_valid > 0
      return anchors[idxs], scores[idxs], box_deltas[idxs]


def class_loss(predicted_scores, y_true):
  # Computes RPN class loss.
  # Parameters
  # predicted_scores : (batch_size, height, width, num_anchors).
  # y_true : (batch_size, height, width, num_anchors, 6).

  epsilon = 1e-7

  # y_true_class: (batch_size, height, width, num_anchors), same as predicted_scores
  y_true_class = y_true[:,:,:,:,1].reshape(predicted_scores.shape)
  y_predicted_class = predicted_scores

  # y_mask: y_true[:,:,:,0] is 1.0 for anchors included in the mini-batch
  y_mask = y_true[:,:,:,:,0].reshape(predicted_scores.shape)

  # Compute how many anchors are actually used in the mini-batch (e.g.,
  # typically 256)
  N_cls = t.count_nonzero(y_mask) + epsilon

  # Compute element-wise loss for all anchors
  loss_all_anchors = F.binary_cross_entropy(input = y_predicted_class, target = y_true_class, reduction = "none")

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors

  # Sum the total loss and normalize by the number of anchors used
  return t.sum(relevant_loss_terms) / N_cls

def regression_loss(predicted_box_deltas, y_true):
  # Computes RPN box delta regression loss.
  # Parameters
  # predicted_box_deltas: (batch_size, height, width, num_anchors * 4) containing
  #   RoI box delta regressions for each anchor, stored as: ty, tx, th, tw.
  # y_true : torch.Tensor (batch_size, height, width, num_anchors, 6).


  epsilon = 1e-7
  scale_factor = 1.0
  sigma = 3.0         
  sigma_squared = sigma * sigma

  y_predicted_regression = predicted_box_deltas
  y_true_regression = y_true[:,:,:,:,2:6].reshape(y_predicted_regression.shape)

  # Include only anchors that are used in the mini-batch and which correspond
  # to objects (positive samples)
  y_included = y_true[:,:,:,:,0].reshape(y_true.shape[0:4]) # trainable anchors map: (batch_size, height, width, num_anchors)
  y_positive = y_true[:,:,:,:,1].reshape(y_true.shape[0:4]) # positive anchors
  y_mask = y_included * y_positive
  y_mask = y_mask.repeat_interleave(repeats = 4, dim = 3)
  N_cls = t.count_nonzero(y_included) + epsilon

  # Compute element-wise loss using robust L1 function for all 4 regression
  # components
  x = y_true_regression - y_predicted_regression
  x_abs = t.abs(x)
  is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * t.sum(relevant_loss_terms) / N_cls
