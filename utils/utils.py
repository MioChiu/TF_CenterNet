import cv2
import numpy as np
import math

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def py_nms(boxes, scores, max_boxes=80, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]

def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=np.float32)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def post_process(detections, org_img_shape, input_size, down_ratio, score_threshold):
    bboxes = detections[0, :, 0:4]
    scores = detections[0, :, 4]
    classes = detections[0, :, 5]
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size[1] / org_w, input_size[0] / org_h)

    dw = (input_size[1] - resize_ratio * org_w) / 2
    dh = (input_size[0] - resize_ratio * org_h) / 2

    bboxes[:, 0::2] = 1.0 * (bboxes[:, 0::2] * down_ratio - dw) / resize_ratio
    bboxes[:, 1::2] = 1.0 * (bboxes[:, 1::2] * down_ratio - dh) / resize_ratio
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, org_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, org_h)
    score_mask = scores >= score_threshold
    bboxes, socres, classes = bboxes[score_mask], scores[score_mask], classes[score_mask]
    return np.concatenate([bboxes, socres[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def bboxes_draw_on_img(img, classes_id, scores, bboxes, class_names, thickness=2):
    colors_tableau = [(158, 218, 229), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207)]
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors_tableau[int(classes_id[i])]
        # Draw bounding boxes
        x1_src = int(bbox[0])
        y1_src = int(bbox[1])
        x2_src = int(bbox[2])
        y2_src = int(bbox[3])

        cv2.rectangle(img, (x1_src, y1_src), (x2_src, y2_src), color, thickness)
        # Draw text
        s = '%s: %.2f' % (class_names[int(classes_id[i])], scores[i])
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (y1_src - text_size[1], x1_src)

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img
