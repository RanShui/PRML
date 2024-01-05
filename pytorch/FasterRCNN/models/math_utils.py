# Math helper functions.


import numpy as np
import torch


def intersection_over_union(boxes1, boxes2):
  # Computes IoU of multiple boxes in gt and to value the effect.
  # boxes1 : np.ndarray
  #   Box corners, shaped (N, 4).
  # boxes2 : np.ndarray
  #   Box corners, shaped (M, 4).
  # Return np.ndarray: IoUs for each pair of boxes in boxes1 and boxes2, shaped (N, M).

  top_left_point = np.maximum(boxes1[:,None,0:2], boxes2[:,0:2])                                  
  # (N,1,2) and (M,2) -> (N,M,2) indicating top-left corners of box pairs
  bottom_right_point = np.minimum(boxes1[:,None,2:4], boxes2[:,2:4])                              
  # "" bottom-right corners ""
  well_ordered_mask = np.all(top_left_point < bottom_right_point, axis = 2)                       
  # (N,M) indicating whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
  intersection_areas = well_ordered_mask * np.prod(bottom_right_point - top_left_point, axis = 2) 
  # (N,M) indicating intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
  areas1 = np.prod(boxes1[:,2:4] - boxes1[:,0:2], axis = 1)                                       
  # (N,) indicating areas of boxes1
  areas2 = np.prod(boxes2[:,2:4] - boxes2[:,0:2], axis = 1)                                       
  # (M,) indicating areas of boxes2
  union_areas = areas1[:,None] + areas2 - intersection_areas                                      
  # (N,1) + (M,) - (N,M) = (N,M), union areas of both boxes
  epsilon = 1e-7
  return intersection_areas / (union_areas + epsilon)


def t_intersection_over_union(boxes1, boxes2):
  # Equivalent to intersection_over_union(), operating on PyTorch tensors.

  # Parameters
  # boxes1 : torch.Tensor
  #   Box corners, shaped (N, 4), with each box as (y1, x1, y2, x2).
  # boxes2 : torch.Tensor
  #   Box corners, shaped (M, 4).

  # Return torch.Tensor to storeIoUs for each pair of boxes in boxes1 and boxes2, shaped (N, M).

  top_left_point = torch.maximum(boxes1[:,None,0:2], boxes2[:,0:2])                                 
  # (N,1,2) and (M,2) -> (N,M,2) indicating top-left corners of box pairs
  bottom_right_point = torch.minimum(boxes1[:,None,2:4], boxes2[:,2:4])                             
  # "" bottom-right corners ""
  well_ordered_mask = torch.all(top_left_point < bottom_right_point, axis = 2)                      
  # (N,M) indicating whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
  intersection_areas = well_ordered_mask * torch.prod(bottom_right_point - top_left_point, dim = 2) 
  # (N,M) indicating intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
  areas1 = torch.prod(boxes1[:,2:4] - boxes1[:,0:2], dim = 1)                                       
  # (N,) indicating areas of boxes1
  areas2 = torch.prod(boxes2[:,2:4] - boxes2[:,0:2], dim = 1)                                       
  # (M,) indicating areas of boxes2
  union_areas = areas1[:,None] + areas2 - intersection_areas                                    
  # (N,1) + (M,) - (N,M) = (N,M), union areas of both boxes
  epsilon = 1e-7
  return intersection_areas / (union_areas + epsilon)



def convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
  # Converts box deltas (ty, tx, th, tw) to boxes (y1, x1, y2, x2).
  # box_deltas : (N, 4). Each row is (ty, tx, th, tw)
  # anchors :  (N, 4) each row being (center_y, center_x, height, width)
  # box_delta_means : (4,)
  # box_delta_stds : (4,)
  # Return np.ndarray: Box coordinates, (N, 4), with each row being (y1, x1, y2, x2).

  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2]  # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
  size = anchors[:,2:4] * np.exp(box_deltas[:,2:4])             # width = anchor_width * exp(tw), height = anchor_height * exp(th)
  boxes = np.empty(box_deltas.shape)
  boxes[:,0:2] = center - 0.5 * size                            # y1, x1
  boxes[:,2:4] = center + 0.5 * size                            # y2, x2
  return boxes


def t_convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):

  # Equivalent to convert_deltas_to_boxes(), operating on PyTorch tensors.
  # Parameters
  # box_deltas : (N, 4). Each row is (ty, tx, th, tw).
  # anchors : (N, 4). Each row is (center_y, center_x, height, width).
  # box_delta_means : (4,).
  # box_delta_stds : (4,). 
  # Return torch.Tensor (N, 4). Each row being (y1, x1, y2, x2).

  box_deltas = box_deltas * box_delta_stds + box_delta_means
  center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2]  
  size = anchors[:,2:4] * torch.exp(box_deltas[:,2:4])              

  boxes = torch.empty(box_deltas.shape, dtype = torch.float32, device = "cuda")
  boxes[:,0:2] = center - 0.5 * size       #y1, x1
  boxes[:,2:4] = center + 0.5 * size       #y2, x2
  return boxes
