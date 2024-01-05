# Classes for computing training-related statistics.

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .models.math_utils import intersection_over_union


class TrainingStatistics:
  def __init__(self):
    (
        self.rpn_class_loss,
        self.rpn_regression_loss,
        self.detector_class_loss,
        self.detector_regression_loss,
    ) = float("inf"), float("inf"), float("inf"), float("inf")
    
    (
        self._rpn_class_losses,
        self._rpn_regression_losses,
        self._detector_class_losses,
        self._detector_regression_losses,
    ) = [], [], [], []


  def on_training_step(self, loss):
    # get all the losses from the loss object and append them to the respective lists
    self._rpn_class_losses.append(loss.rpn_class)
    self._rpn_regression_losses.append(loss.rpn_regression)
    self._detector_class_losses.append(loss.detector_class)
    self._detector_regression_losses.append(loss.detector_regression)
    self.rpn_class_loss = np.mean(self._rpn_class_losses)
    self.rpn_regression_loss = np.mean(self._rpn_regression_losses)
    self.detector_class_loss = np.mean(self._detector_class_losses)
    self.detector_regression_loss = np.mean(self._detector_regression_losses)


  def get_progbar_postfix(self):
    # Return Dict[str, str] for a tqdm progress bar.
    return { 
      "detector_class_loss": "%1.4f" % self.detector_class_loss,
      "detector_regr_loss": "%1.4f" % self.detector_regression_loss,
      "rpn_class_loss": "%1.4f" % self.rpn_class_loss,
      "rpn_regr_loss": "%1.4f" % self.rpn_regression_loss,
      "total_loss": "%1.2f" % (self.rpn_class_loss + self.rpn_regression_loss + self.detector_class_loss + self.detector_regression_loss)
    }
  
  def get_wandb_log(self, iteration):
    # Return Dict[str, str] for a tqdm progress bar.
    return { 
      "iteration": iteration,
      "detector_class_loss": self.detector_class_loss.item(),
      "detector_regr_loss": self.detector_regression_loss.item(),
      "rpn_class_loss": self.rpn_class_loss.item(),
      "rpn_regr_loss": self.rpn_regression_loss.item(),
      "total_loss": (self.rpn_class_loss.item() + self.rpn_regression_loss.item() + self.detector_class_loss.item() + self.detector_regression_loss.item())
    }
  


class PrecisionRecallCurveCalculator:
  # Computes precision and recall (including mean average precision).

  def __init__(self):
    # List and true number of (confidence_score, correctness) by class
    self._unsorted_predictions_by_class_index = defaultdict(list)
    self._object_count_by_class_index = defaultdict(int)


  def _correctness_of_predictions(self, scored_boxes_by_class_index, gt_boxes):
    unsorted_predictions_by_class_index = {}
    object_count_by_class_index = defaultdict(int)

    # Class balancing strategy.
    for gt_box in gt_boxes:
      object_count_by_class_index[gt_box.class_index] += 1

    for class_index, scored_boxes in scored_boxes_by_class_index.items():
      gt_boxes_this_class = [ gt_box for gt_box in gt_boxes if gt_box.class_index == class_index ]

      # Compute IoU of each box with gt and store as a list of tuple[iou, box_index, gt_box_index]
      ious = []
      boxes = len(gt_boxes_this_class)
      s_boxes = len(scored_boxes)

      for gt_idx in range(boxes):
        for box_idx in range(s_boxes):
          boxes1 = np.expand_dims(scored_boxes[box_idx][0:4], axis = 0) 
          # convert single box (4,) to (1,4) for broadcasting
          boxes2 = np.expand_dims(gt_boxes_this_class[gt_idx].corners, axis = 0)
          iou = intersection_over_union(boxes1 = boxes1, boxes2 = boxes2) 
          ious.append((iou, box_idx, gt_idx))

      ious = sorted(ious, key = lambda iou: ious[0], reverse = True)  # descending sorting by IoU
      

      # whether a gt has been detected
      gt_box_detected = [ False ] * len(gt_boxes)


      # whether a prediction is a true+ (True) or false+ (False)
      is_true_positive = [ False ] * len(scored_boxes)
      
      
      # Construct a list of prediction descriptions: (score, correct)
      # Images with IoU <= 0.5 or without the highest IoU for any gt box are considered false+.
      
      iou_threshold = 0.5
      for iou, box_idx, gt_idx in ious:

        if iou <= iou_threshold:
          continue
        if is_true_positive[box_idx] or gt_box_detected[gt_idx]:
          # The prediction or gt box hav been matched
          continue

        # Got a true positive
        is_true_positive[box_idx] = True
        gt_box_detected[gt_idx] = True
      # Construct the final array of prediction descriptions
      unsorted_predictions_by_class_index[class_index] = [ (scored_boxes[i][4], is_true_positive[i]) for i in range(len(scored_boxes)) ]
        
    return unsorted_predictions_by_class_index, object_count_by_class_index


  def add_image_results(self, scored_boxes_by_class_index, gt_boxes):

    # scored_boxes_by_class_index : Final detected boxes as lists of tuple[y_min, x_min, y_max, x_max, score] by class index.
    # Merge in results for this single image
    unsorted_predictions_by_class_index, object_count_by_class_index = self._correctness_of_predictions(scored_boxes_by_class_index = scored_boxes_by_class_index, gt_boxes = gt_boxes) 

    for class_index, predictions in unsorted_predictions_by_class_index.items():
      self._unsorted_predictions_by_class_index[class_index] += predictions
    for class_index, count in object_count_by_class_index.items():
      self._object_count_by_class_index[class_index] += object_count_by_class_index[class_index]


  def _compute_average_precision(self, class_index):
    # Sort predictions in descending order of score
    sorted_predictions = sorted(self._unsorted_predictions_by_class_index[class_index], key = lambda prediction: prediction[0], reverse = True)
    num_ground_truth_positives = self._object_count_by_class_index[class_index]

    # Compute raw recall and precision arrays
    recall_array = []
    precision_array = []
    true_positives = 0  # running tally
    false_positives = 0 # ""
    for i in range(len(sorted_predictions)):
      true_positives += 1 if sorted_predictions[i][1] == True else 0
      false_positives += 0 if sorted_predictions[i][1] == True else 1

      recall = true_positives / num_ground_truth_positives
      precision = true_positives / (true_positives + false_positives)
      recall_array.append(recall)
      precision_array.append(precision)

    # Insert 0 at the beginning and end. 
    recall_array.insert(0, 0.0)
    recall_array.append(1.0)
    precision_array.insert(0, 0.0)
    precision_array.append(0.0)

    # Compute interpolated precision
    precision = len(precision_array)
    for i in range(precision):
      precision_array[i] = np.max(precision_array[i:])
    
    # Compute AP using simple rectangular integration under the curve
    average_precision = 0
    recall = len(recall_array) - 1
    for i in range(recall):
      dx = recall_array[i + 1] - recall_array[i + 0]
      dy = precision_array[i + 1]
      average_precision = average_precision + dx * dy

    return average_precision, recall_array, precision_array



  def compute_mean_average_precision(self):
    # Calculates mAP, only after all image results have been processed.
    # Return np.float64: Mean average precision.

    average_precisions = []
    for class_index in self._object_count_by_class_index:
      average_precision, _, _ = self._compute_average_precision(class_index = class_index)
      average_precisions.append(average_precision)
    return np.mean(average_precisions)
  

  def plot_precision_vs_recall(self, class_index, class_name = None, interpolated = False):
    # Plots precision (y axis) vs. recall (x axis), only after all image results have been processed.
    # class_index : The class index for the curve.
    # class_name : Used on the plot label.

    average_precision, recall_array, precision_array = self._compute_average_precision(class_index = class_index, interpolated = interpolated)


    # Plot raw precision vs. recall
    if class_name is None:
      label = "Class {} AP={:1.2f}".format(class_index, average_precision)
    else:
      label = "{} AP={:1.2f}".format(class_name, average_precision)

    plt.plot(recall_array, precision_array, label = label)
    if interpolated:
      plt.title("Precision (Interpolated) vs. Recall")
    else:
      plt.title("Precision vs. Recall")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
    plt.clf()



  def plot_average_precisions(self, class_index_to_name): 
    labels = [class_index_to_name[class_index] for class_index in self._object_count_by_class_index]

    average_precisions = []
    for class_index in self._object_count_by_class_index:
      average_precision, _, _ = self._compute_average_precision(class_index = class_index)
      average_precisions.append(average_precision)

    # Sort alphabetically by class name
    sorted_results = sorted(zip(labels, average_precisions), reverse = True, key = lambda pair: pair[0])
    labels, average_precisions = zip(*sorted_results) 
    
    # Convert to percentage
    average_precisions = np.array(average_precisions) * 100.0 

    # Bar plot
    plt.clf()
    plt.xlim([0, 100])
    plt.barh(labels, average_precisions)
    plt.title("Model Performance")
    plt.xlabel("Average Precision (%)")
    for index, value in enumerate(average_precisions):
      plt.text(value, index, "%1.1f" % value)
    plt.show()


  def print_average_precisions(self, class_index_to_name):
    # Compute average precisions for each class
    labels = [ class_index_to_name[class_index] for class_index in self._object_count_by_class_index ]
    average_precisions = []
    for class_index in self._object_count_by_class_index:
      average_precision, _, _ = self._compute_average_precision(class_index = class_index)
      average_precisions.append(average_precision)

    # Sort by score (descending)
    sorted_results = sorted(zip(labels, average_precisions), reverse = True, key = lambda pair: pair[1])
    _, average_precisions = zip(*sorted_results)


    label_width = max([len(label) for label in labels])

    # Pretty print
    print("Average Precisions")
    print("***********************************")
    for (label, average_precision) in sorted_results:
      print("%s: %1.1f%%" % (label.ljust(label_width), average_precision * 100.0))
    print("***********************************")
