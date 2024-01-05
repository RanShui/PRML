# Backbone base class for wrapping backbone models that provide feature
# extraction and pooled feature reduction layers from the classifier
# stages. The backbone is used:
#   1. as the feature extractor. 
#   2. as the detector


import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from ..Image_Processing import image


class Backbone:

  
  def __init__(self):
    self.feature_map_channels = 0     # feature map channels
    self.feature_pixels = 0           # feature size in pixels, N: each feature map cell corresponds to an NxN area on original image
    self.feature_vector_size = 0      # length of linear feature vector after pooling and just before being passed to detector heads
    self.image_preprocessing_params = image.Preprocessing(channel_order = image.ColorOrder.BGR, scaling = 1.0, means = [ 103.939, 116.779, 123.680 ], stds = [ 1, 1, 1 ])

    # Required members
    self.feature_extractor = None       
    self.pool_to_feature_vector = None  



  def compute_feature_map_shape(self, image_shape):
    # Input image_shape : Tuple[int, int, int]
    # Return Tuple[feature_map_channels, feature_map_height, feature_map_width].
    return image_shape[-3:]