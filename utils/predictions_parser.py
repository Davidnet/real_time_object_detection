from .category_index_coco import CATEGORY_INDEX
from collections import namedtuple
from phi import P, Obj, Rec
import cytoolz as cz
import numpy as np


def create_pred(tup, size):
    name, score, box = tup
    name = CATEGORY_INDEX[name]
    distance = distance_approximator(box)
    if size is not None:
        box = pixel_coordinates(box, size)
    return dict(name=name["name"], score=score, box=box, id=name['id'], distance=distance)

def pixel_coordinates(box, size):
    # SHAPES ARE INVERTED
    ymin, xmin, ymax, xmax = box
    return (ymin*(size[0]-1), xmin*(size[1]-1), ymax*(size[0]-1), xmax*(size[1]-1))


def distance_approximator(box):
    """ Function that does the distance approximation """
    ymin, xmin, ymax, xmax = box
    mid_x = (xmax + xmin) / 2
    mid_y = (ymax + ymin) / 2  # TODO: use mid_y
    apx_distance = round((1 - (xmax - xmin)) ** 4, 1)
    return apx_distance

def detection_list(boxes, classes, scores, max_boxes_to_report=None, min_score_thresh=.5):
    """Create dictionary of detections"""
    detections = list()
    if not max_boxes_to_report:
        max_boxes_to_report = len(boxes)
    for i in range(min(max_boxes_to_report, len(boxes))):
        if scores is None or scores[i] > min_score_thresh:

            class_name = CATEGORY_INDEX[classes[i]]['name']
            box = tuple(boxes[i].tolist())
            score = int(100*scores[i])
            detections.append((class_name, {"score": score, "coordinates": box}))
    return detections


def parser(num, boxes, scores, classes, **kargs):

    min_score = kargs.pop("min_score", 0.5)
    max_predictions = kargs.pop("max_predictions", 20)
    normalized_predictions = kargs.pop("normalized_predictions", True) #By default FALSE TODO: 
    image_shape = kargs.pop("image_shape", None)

    predictions = dict(num_detections = num,
                boxes  =  boxes, 
                scores =  scores,
                classes = classes)

    predictions['num_detections'] = int(predictions['num_detections'][0].tolist())
    predictions['classes'] = predictions[
        'classes'][0].astype(np.uint8).tolist()
    predictions['boxes'] = predictions['boxes'][0].tolist()
    predictions['scores'] = predictions['scores'][0].tolist()
    predictions = zip(predictions['classes'], predictions['scores'], predictions['boxes'])
    if normalized_predictions:
        predictions = map(lambda tup: create_pred(tup, None), predictions)
    else:
        predictions = map(lambda tup: create_pred(tup, image_shape), predictions)
    predictions = filter(P["score"] > min_score, predictions)
    predictions = sorted(predictions, key = P["score"])
    predictions = cz.take(max_predictions, predictions)
    predictions = list(predictions)

    return predictions

