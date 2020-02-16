import tensorflow as tf

def nms(heat, kernel=3):
    hmax = tf.layers.max_pooling2d(heat, kernel, 1, padding='same')
    keep = tf.cast(tf.equal(heat, hmax), tf.float32)
    return heat*keep

def topk(hm, K=100):
    batch, height, width, cat = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    #[b,h*w*c]
    scores = tf.reshape(hm, (batch, -1))
    #[b,k]
    topk_scores, topk_inds = tf.nn.top_k(scores, k=K)
    #[b,k]
    topk_clses = topk_inds % cat
    topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
    topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
    topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs 

def decode(heat, wh, reg=None, K=100):
    batch, height, width, cat = tf.shape(heat)[0], tf.shape(heat)[1], tf.shape(heat)[2], tf.shape(heat)[3]
    heat = nms(heat)
    scores, inds, clses, ys, xs = topk(heat, K=K)

    if reg is not None:
        reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
        #[b,k,2]
        reg = tf.batch_gather(reg, inds)
        xs = tf.expand_dims(xs, axis=-1) + reg[..., 0:1]
        ys = tf.expand_dims(ys, axis=-1) + reg[..., 1:2]
    else:
        xs = tf.expand_dims(xs, axis=-1) + 0.5
        ys = tf.expand_dims(ys, axis=-1) + 0.5
    
    #[b,h*w,2]
    wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
    #[b,k,2]
    wh = tf.batch_gather(wh, inds)

    clses = tf.cast(tf.expand_dims(clses, axis=-1), tf.float32)
    scores = tf.expand_dims(scores, axis=-1)

    xmin = xs - wh[...,0:1] / 2
    ymin = ys - wh[...,1:2] / 2
    xmax = xs + wh[...,0:1] / 2
    ymax = ys + wh[...,1:2] / 2

    bboxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    #[b,k,6]
    detections = tf.concat([bboxes, scores, clses], axis=-1)
    return detections