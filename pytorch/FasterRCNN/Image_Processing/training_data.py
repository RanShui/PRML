# All of the training data bound together of a sample.


import numpy as np
from PIL import Image
from typing import List
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Box:
  class_index: int
  class_name: str
  corners: np.ndarray

@dataclass
class Training:  

  # file path of image
  filepath:                   str 

  # PIL image data (for debug rendering), scaled    
  image:                      Image  
  
  # shape (3,height,width), pre-processed and scaled to expected size    
  image_data:                 np.ndarray            

  # shape (feature_map_height,feature_map_width,num_anchors*4), with each anchor as [center_y,center_x,height,width] 
  anchor_map:                 np.ndarray  

  # shape (feature_map_height,feature_map_width,num_anchors), indicating which anchors are valid (do not cross image boundaries)
  anchor_valid_map:           np.ndarray  

  # torch.Tensor  Ground truth RPN map of shape (batch_size, height, width, num_anchors, 6)             
  gt_rpn_map:                 np.ndarray     

  # list of (y,x,k) coordinates of anchors in gt_rpn_map that are labeled as object        
  gt_rpn_object_indices:      List[Tuple[int,int,int]]  

  # list of (y,x,k) coordinates of background anchors
  gt_rpn_background_indices:  List[Tuple[int,int,int]]  

  # list of ground-truth boxes, scaled
  gt_boxes:                   List[Box]             
