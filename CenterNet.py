import tensorflow as tf
import cfg
import loss
from net import resnet
from net.layers import _conv, upsampling
import numpy as np
class CenterNet():
    def __init__(self, inputs, is_training):
        self.is_training = is_training
        try:
            self.pred_hm, self.pred_wh, self.pred_reg = self._build_model(inputs)
        except:
            raise NotImplementedError("Can not build up centernet network!")

    def _build_model(self, inputs):
        with tf.variable_scope('resnet'):
            c2, c3, c4, c5 = resnet.resnet34(is_training=self.is_training).forward(inputs)

            p5 = _conv(c5, 128, [1,1], is_training=self.is_training)

            up_p5 = upsampling(p5, method='resize')
            reduce_dim_c4 = _conv(c4, 128, [1,1], is_training=self.is_training)
            p4 = 0.5*up_p5 + 0.5*reduce_dim_c4

            up_p4 = upsampling(p4, method='resize')
            reduce_dim_c3 = _conv(c3, 128, [1,1], is_training=self.is_training)
            p3 = 0.5*up_p4 + 0.5*reduce_dim_c3

            up_p3 = upsampling(p3, method='resize')
            reduce_dim_c2 = _conv(c2, 128, [1,1], is_training=self.is_training)
            p2 = 0.5*up_p3 + 0.5*reduce_dim_c2

            features = _conv(p2, 128, [3,3], is_training=self.is_training)

            # IDA-up
            # p2 = _conv(c2, 128, [1,1], is_training=self.is_training)
            # p3 = _conv(c3, 128, [1,1], is_training=self.is_training)
            # p4 = _conv(c4, 128, [1,1], is_training=self.is_training)
            # p5 = _conv(c5, 128, [1,1], is_training=self.is_training)

            # up_p3 = upsampling(p3, method='resize')
            # p2 = _conv(p2+up_p3, 128, [3,3], is_training=self.is_training)

            # up_p4 = upsampling(upsampling(p4, method='resize'), method='resize')
            # p2 = _conv(p2+up_p4, 128, [3,3], is_training=self.is_training)

            # up_p5 = upsampling(upsampling(upsampling(p5, method='resize'), method='resize'), method='resize')
            # features = _conv(p2+up_p5, 128, [3,3], is_training=self.is_training)
        
        with tf.variable_scope('detector'):
            hm = _conv(features, 64, [3,3], is_training=self.is_training)
            hm = tf.layers.conv2d(hm, cfg.num_classes, 1, 1, padding='valid', activation = tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm')

            wh = _conv(features, 64, [3,3], is_training=self.is_training)
            wh = tf.layers.conv2d(wh, 2, 1, 1, padding='valid', activation = None, name='wh')
            
            reg =  _conv(features, 64, [3,3], is_training=self.is_training)
            reg = tf.layers.conv2d(reg, 2, 1, 1, padding='valid', activation = None, name='reg')

        return hm, wh, reg

    def compute_loss(self, true_hm, true_wh, true_reg, reg_mask, ind):
        hm_loss = loss.focal_loss(self.pred_hm, true_hm)
        wg_loss = 0.05*loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask)
        reg_loss = loss.reg_l1_loss(self.pred_reg, true_reg, ind, reg_mask)
        return hm_loss, wg_loss, reg_loss