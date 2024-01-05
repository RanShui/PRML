##Shujie Zhang
##JPEGImage Pre-processing

import imageio
import numpy as np
from enum import Enum
from PIL import Image
from PIL import ImageOps
from typing import List
from dataclasses import dataclass
import torchvision.transforms as transforms

class ColorOrder(Enum):
  RGB = "RGB"
  BGR = "BGR"


@dataclass
class Preprocessing:
  # Image preprocessed parameters. 

  channel_order: ColorOrder
  scaling: float
  means: List[float]
  stds: List[float]


def _compute_scale_factor(original_width, original_height, min_dimension_pixels):
    return 1.0 if not min_dimension_pixels else max(min_dimension_pixels / original_height, min_dimension_pixels / original_width)

def resize_image(image, scale_factor):
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    return image.resize(new_size, resample=Image.BILINEAR)

def load_image(url, preprocessing, min_dimension_pixels = None, horizontal_flip = False):
  
  # Standardizing image pixels, adjusting channel orders(to backbone) 
  # 1 Parameters
  # ----------
  # url : local or remote file to load.
  # preprocessing : Image pre-processing parameters governing channel order and normalization.
  # min_dimension_pixels : Standardize the size of the smaller side of the image.
  # horizontal_flip : Whether to flip the image horizontally.

  # 2 Return
  # np.ndarray: Image pixels float32 (channels, height, width)
  # PIL.Image: an image object suitable for drawing and visualization
  # float: scaling factor 
  # Tuple[int, int, int]: the original image shape (channels, height, width)
  

  data = imageio.imread(url, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  original_width = image.width
  original_height = image.height

  if horizontal_flip:
    image = ImageOps.mirror(image)

  scale_factor = _compute_scale_factor(original_width=image.width, original_height=image.height, min_dimension_pixels=min_dimension_pixels)
  image = resize_image(image, scale_factor)

  image_data = _preprocess(image = image, preprocessing = preprocessing)

  return image_data, image, scale_factor, (image_data.shape[0], original_height, original_width)



def _preprocess(image, preprocessing): 
  image_data = np.array(image).astype(np.float32)

  # RGB -> BGR
  if preprocessing.channel_order not in [ColorOrder.BGR, ColorOrder.RGB]:
    raise ValueError("!Invalid: %s; Color order should be one of RGB or BGR!" % str(preprocessing.channel_order))
  else:
    if preprocessing.channel_order == ColorOrder.BGR:
        image_data = image_data[:, :, ::-1]

  # enumerate three channels
  for channel in range(3):  
    image_data[:, :, channel] *= preprocessing.scaling
    image_data[:, :, channel] = (image_data[:, :, channel] - preprocessing.means[channel]) / preprocessing.stds[channel]
  
  # dimension from (height,width,3) to (3,height,width)
  image_data = np.moveaxis(image_data, -1, 0)  
  # copy required to eliminate negative stride
  return image_data.copy()                      



def _preprocess_resnet(image):
    image_data = np.array(image).astype(np.float32)
    # 转换为 PyTorch 张量并归一化
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_data = preprocess(image_data)
    return image_data.copy()