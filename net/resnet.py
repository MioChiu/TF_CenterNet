import tensorflow as tf
import numpy as np
from net.layers import _conv



class ResNet():
    def __init__(self, arch, layers, base_filters=64, is_training=False, use_bn=True):
        if arch == 'resnet18' or arch == 'resnet34':
            self.block = self.BasicBlock
        elif arch == 'resnet50' or arch =='resnet101':
            self.block = self.Bottleneck
        else:
            raise ValueError('only support resnet18 34 50 101')
        self.layers = layers
        self.base_filters = base_filters
        if self.block == self.BasicBlock:
            assert self.base_filters == 64
        self.is_training = is_training 
        self.use_bn = use_bn
        self.inplanes = 64

    def BasicBlock(self, inputs, filters, strides=1, is_training=False, use_bn=True):
        expansion = 1
        conv1_bn_relu = _conv(inputs, filters, [3,3], strides, 'same', activation=tf.nn.relu, is_training=self.is_training, use_bn=self.use_bn)
        conv2_bn = _conv(conv1_bn_relu, filters, [3,3], 1, 'same', activation=None, is_training=self.is_training, use_bn=self.use_bn)
        if strides != 1 or self.inplanes != filters*expansion:
            inputs = _conv(inputs, filters, [1,1], strides, 'valid', activation=None, is_training=self.is_training, use_bn=self.use_bn)
            self.inplanes = filters*expansion
        out = tf.nn.relu(conv2_bn+inputs)
        return out

    def Bottleneck(self, inputs, filters, strides=1, is_training=False, use_bn=True):
        expansion = 4
        conv1_bn_relu = _conv(inputs, filters, [1,1], 1, 'valid', activation=tf.nn.relu, is_training=self.is_training, use_bn=self.use_bn)
        conv2_bn_relu = _conv(conv1_bn_relu, filters, [3,3], strides, 'same', activation=tf.nn.relu, is_training=self.is_training, use_bn=self.use_bn)
        conv3_bn = _conv(conv2_bn_relu, filters*expansion, [1,1], 1, 'valid', activation=None, is_training=self.is_training, use_bn=self.use_bn)
        if strides != 1 or self.inplanes != filters*expansion:
            inputs = _conv(inputs, filters*expansion, [1,1], strides, 'valid', activation=None, is_training=self.is_training, use_bn=self.use_bn)
            self.inplanes = filters*expansion
        out = tf.nn.relu(conv3_bn+inputs)
        return out

    def _make_layer(self, x, num_channels, layers, strides=1):
        for i in range(layers):
            if i == 0:
                x = self.block(x, num_channels, strides=strides)
            else:
                x = self.block(x, num_channels)
        return x
    
    def _layer0(self, inputs, filters, kernel_size=(7, 7)):
        outputs = _conv(inputs, filters, [7,7],  2, 'same',  activation=tf.nn.relu, is_training=self.is_training, use_bn=self.use_bn)
        outputs = tf.layers.max_pooling2d(outputs, pool_size=3, strides=2, padding='same')
        return outputs

    def forward(self, inputs):
        self.layer0 = self._layer0(inputs, self.inplanes, (7, 7))
        self.layer1 = self._make_layer(self.layer0, self.base_filters, self.layers[0])
        self.layer2 = self._make_layer(self.layer1, 2 * self.base_filters, self.layers[1], 2)
        self.layer3 = self._make_layer(self.layer2, 4 * self.base_filters, self.layers[2], 2)
        self.layer4 = self._make_layer(self.layer3, 8 * self.base_filters, self.layers[3], 2)
        
        return self.layer1, self.layer2, self.layer3,self.layer4
            
def load_weights(sess,path):
    pretrained = np.load(path,allow_pickle=True).item()
    for variable in tf.trainable_variables():
        for key in pretrained.keys():
            key2 = variable.name.rstrip(':0')
            if (key == key2):
                sess.run(tf.assign(variable, pretrained[key])) 


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(**kwargs):
    return _resnet('resnet18', [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return _resnet('resnet34', [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return _resnet('resnet50', [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return _resnet('resnet101', [3, 4, 23, 3], **kwargs)

if __name__ =='__main__':
    inputs = tf.placeholder(shape=[None,300,300,3],dtype=tf.float32)
    net = resnet18(is_training=True).forward(inputs)
    for variable in tf.trainable_variables():
        print(variable.name,variable.shape)