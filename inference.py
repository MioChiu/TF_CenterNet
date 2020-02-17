import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import time
import cfg
from CenterNet import CenterNet
from utils.decode import decode
from utils.image import get_affine_transform, affine_transform
from utils.utils import image_preporcess, py_nms, post_process, bboxes_draw_on_img, read_class_names

ckpt_path='./checkpoint/'
sess = tf.Session()

inputs = tf.placeholder(shape=[None,None,None,3],dtype=tf.float32)
model = CenterNet(inputs, False)
saver = tf.train.Saver()
saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))

hm = model.pred_hm
wh = model.pred_wh
reg = model.pred_reg
det = decode(hm, wh, reg, K=cfg.max_objs)

class_names= read_class_names(cfg.classes_file)
img_names = os.listdir('D:/dataset/VOC/test/VOCdevkit/VOC2007/JPEGImages')
for img_name in img_names:
    img_path = 'D:/dataset/VOC/test/VOCdevkit/VOC2007/JPEGImages/' + img_name
    original_image = cv2.imread(img_path)
    original_image_size = original_image.shape[:2]
    image_data = image_preporcess(np.copy(original_image), [cfg.input_image_h, cfg.input_image_w])
    image_data = image_data[np.newaxis, ...]

    t0 = time.time()
    detections = sess.run(det, feed_dict={inputs: image_data})
    detections = post_process(detections, original_image_size, [cfg.input_image_h,cfg.input_image_w], cfg.down_ratio, cfg.score_threshold)
    print('Inferencce took %.1f ms (%.2f fps)' % ((time.time()-t0)*1000, 1/(time.time()-t0)))
    if cfg.use_nms:
        cls_in_img = list(set(detections[:,5]))
        results = []
        for c in cls_in_img:
            cls_mask = (detections[:,5] == c)
            classified_det = detections[cls_mask]
            classified_bboxes = classified_det[:, :4]
            classified_scores = classified_det[:, 4]
            inds = py_nms(classified_bboxes, classified_scores, max_boxes=50, iou_thresh=0.5)
            results.extend(classified_det[inds])
        results = np.asarray(results)
        if len(results) != 0:
            bboxes = results[:,0:4]
            scores = results[:,4]
            classes = results[:, 5]
            bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)
        
    else:
        bboxes = detections[:,0:4]
        scores = detections[:,4]
        classes = detections[:,5]
        bboxes_draw_on_img(original_image, classes, scores, bboxes, class_names)

    cv2.imshow('img',original_image)
    cv2.waitKey()
