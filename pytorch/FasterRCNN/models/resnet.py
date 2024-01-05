# PyTorch implementation of the ResNet backbone for use as a feature extractor in Faster R-CNN. 

import torch as t
import torchvision
import numpy as np
import torchvision.transforms as transforms
from enum import Enum
from math import ceil
from torch import nn
from torch.nn import functional as F
from ..Image_Processing import image
from .backbone import Backbone


class Architecture(Enum):
  ResNet50 = "ResNet50"
  ResNet101 = "ResNet101"


class FeatureExtractor(nn.Module):

  def __init__(self, resnet):
    super().__init__()

    # Feature extractor layers
    self._feature_extractor = nn.Sequential(
      resnet.conv1,     # 0
      resnet.bn1,       # 1
      nn.GELU(),        # 2 Variable
      resnet.maxpool,   # 3
      resnet.layer1,    # 4
      resnet.layer2,    # 5
      resnet.layer3     # 6
    )

    # Batchnorm layers does not need to be changed
    self._freeze_batchnorm(self._feature_extractor)

  


  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)
        
  def forward(self, image_data):
    return self._feature_extractor(image_data)

class PoolToFeatureVector(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4
    self._freeze_batchnorm(self._layer4)

  def train(self, mode = True):
    # See comments in FeatureVector.train()
    super().train(mode)
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._layer4.apply(set_bn_eval)

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)

    # Average together the last two dimensions to remove them -> (N, 2048).
    # It is also possible to max pool, e.g.:
    # y = F.adaptive_max_pool2d(y, output_size = 1).squeeze()
    # This may even be better (74.96% mAP for ResNet50 vs. 73.2% using the
    # current method).
    y = y.mean(-1).mean(-1) # use mean to remove last two dimensions -> (N, 2048)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)


class ResNetBackbone(Backbone):
  def __init__(self, architecture):
    super().__init__()

    # Backbone properties. Image preprocessing parameters are common to all
    # Torchvision ResNet models and are described in the documentation, e.g.,
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    self.feature_map_channels = 1024  # feature extractor output channels
    self.feature_pixels = 16          # ResNet feature maps are 1/16th of the original image size, similar to VGG-16 feature extractor
    self.feature_vector_size = 2048   # linear feature vector size after pooling
    self.image_preprocessing_params = image.Preprocessing(channel_order = image.ColorOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])

    # Construct model and pre-load with ImageNet weights
    if architecture == Architecture.ResNet50:
      resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
      # not pretrain model
      # resnet = torchvision.models.resnet50(pretrained=False)
    elif architecture == Architecture.ResNet101:
      resnet = torchvision.models.resnet101(weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
    elif architecture == Architecture.ResNet152:
      resnet = torchvision.models.resnet152(weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    else:
      raise ValueError("Invalid ResNet architecture value: %s" % architecture.value)
    print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision %s backbone" % architecture.value)

    # Feature extractor: given image data of shape (batch_size, channels, height, width),
    # produces a feature map of shape (batch_size, 1024, ceil(height/16), ceil(width/16))
    self.feature_extractor = FeatureExtractor(resnet = resnet)

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(resnet = resnet)

  def compute_feature_map_shape(self, image_shape):
    # ResNet convolutional layers use padding and the dimensions are not simply 
    # an integral division by 16 like VGG-16. The calculation of self.feature_pixels works well by trial.
    # Parameters image_shape : Tuple[channels, height, width]
    # Return Tuple[feature_map_channels, feature_map_height, feature_map_width].
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))


def _preprocess_resnet(image):
    image_data = np.array(image).astype(np.float32)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_data = preprocess(image_data)
    return image_data.copy()