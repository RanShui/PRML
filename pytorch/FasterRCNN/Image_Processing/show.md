``` 
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
```