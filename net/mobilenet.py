import tensorflow as tf
# import tensorflow.contrib as tc

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_bn_relu(inputs, filters, kernel_size=3, strides=1, is_training=False, is_depthwise=False):
    if is_depthwise:
        outputs = tf.contrib.layers.separable_conv2d(
            inputs,            
            None, 
            kernel_size,
            depth_multiplier=1, 
            stride=(strides, strides),
            padding='SAME',
            activation_fn=None,
            weights_initializer=tf.glorot_uniform_initializer(),
            biases_initializer=None)
        outputs = tf.layers.batch_normalization(
            inputs=outputs,
            training=is_training,
            momentum = 0.95
        )
        outputs = tf.nn.relu6(outputs)
    
    else:
        outputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias = False
        )
        outputs = tf.layers.batch_normalization(
            inputs=outputs,
            training=is_training,
            momentum = 0.95
        )
        outputs = tf.nn.relu6(outputs)
    
    return outputs
        
def _inverted_residual(inputs, channels, strides, expand_ratio, is_training=False):
    assert strides in [1,2]
    inp = inputs.get_shape().as_list()[-1]
    oup = channels

    hidden_dim = int(round(inp * expand_ratio))
    use_res_connect = strides == 1 and inp == oup
    outputs = inputs
    if expand_ratio != 1:
        outputs = _conv_bn_relu(outputs, hidden_dim, kernel_size=1, is_training=is_training)
    
    outputs = _conv_bn_relu(outputs, hidden_dim, strides=strides, is_training=is_training, is_depthwise=True)
    outputs = tf.layers.conv2d(outputs, oup, 1, 1, use_bias=False)
    outputs = tf.layers.batch_normalization(inputs=outputs, training=is_training, momentum = 0.95)

    if use_res_connect:
        outputs = tf.add(inputs, outputs)
    return outputs

class MobileNetV2():
    def __init__(self,
                 inputs,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 is_training=False):

        self.inputs = inputs
        self.width_mult = width_mult
        self.inverted_residual_setting = inverted_residual_setting
        self.round_nearest = round_nearest
        self.is_training = is_training

        self.input_channel = 32
        self.last_channel = 1280

        if self.inverted_residual_setting is None:
            self.inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],  #1/2
                [6, 24, 2, 2],  #1/4
                [6, 32, 3, 2],  #1/8
                [6, 64, 4, 2],  #1/16
                [6, 96, 3, 1],  #1/16
                [6, 160, 3, 2], #1/32
                [6, 320, 1, 1], #1/32
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(self.inverted_residual_setting) == 0 or len(self.inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))


    def forward(self):
        # building first layer
        self.input_channel = _make_divisible(self.input_channel * self.width_mult, self.round_nearest)
        self.last_channel = _make_divisible(self.last_channel * max(1.0, self.width_mult), self.round_nearest)
        features = _conv_bn_relu(self.inputs, self.input_channel, strides=2, is_training=self.is_training)
        # building inverted residual blocks
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = _make_divisible(c * self.width_mult, self.round_nearest)
            for i in range(n):
                strides = s if i == 0 else 1
                features = _inverted_residual(features, output_channel, strides, expand_ratio=t, is_training=self.is_training)
                self.input_channel = output_channel
            
            if c == 32:
                self.layer1 = features
            elif c == 96:
                self.layer2 = features
            elif c == 320:
                self.layer3 = features


        # building last several layers
        self.features = _conv_bn_relu(features, self.last_channel, kernel_size=1, is_training=self.is_training)

        return self.layer1, self.layer2, self.layer3


def mobilenet_v2(**kwargs):
    model = MobileNetV2(**kwargs)
    return model

if __name__=='__main__':
    inputs = tf.placeholder(shape=[None,300,300,3],dtype=tf.float32)
    net = mobilenet_v2(inputs=inputs,is_training=True).forward()
    for variable in tf.trainable_variables():
        print(variable.name[:-2], variable.shape)
