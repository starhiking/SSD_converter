import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    data            = tf.placeholder(tf.float32, shape = (None, 300, 300, 3), name = 'data')
    conv0_pad       = tf.pad(data, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv0           = convolution(conv0_pad, group=1, strides=[2, 2], padding='VALID', name='conv0')
    conv0_bn        = batch_normalization(conv0, variance_epsilon=9.999999747378752e-06, name='conv0/bn')
    conv0_relu      = tf.nn.relu(conv0_bn, name = 'conv0/relu')
    conv1_dw_pad    = tf.pad(conv0_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv1_dw        = convolution(conv1_dw_pad, group=32, strides=[1, 1], padding='VALID', name='conv1/dw')
    conv1_dw_bn     = batch_normalization(conv1_dw, variance_epsilon=9.999999747378752e-06, name='conv1/dw/bn')
    conv1_dw_relu   = tf.nn.relu(conv1_dw_bn, name = 'conv1/dw/relu')
    conv1           = convolution(conv1_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv1')
    conv1_bn        = batch_normalization(conv1, variance_epsilon=9.999999747378752e-06, name='conv1/bn')
    conv1_relu      = tf.nn.relu(conv1_bn, name = 'conv1/relu')
    conv2_dw_pad    = tf.pad(conv1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv2_dw        = convolution(conv2_dw_pad, group=64, strides=[2, 2], padding='VALID', name='conv2/dw')
    conv2_dw_bn     = batch_normalization(conv2_dw, variance_epsilon=9.999999747378752e-06, name='conv2/dw/bn')
    conv2_dw_relu   = tf.nn.relu(conv2_dw_bn, name = 'conv2/dw/relu')
    conv2           = convolution(conv2_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv2')
    conv2_bn        = batch_normalization(conv2, variance_epsilon=9.999999747378752e-06, name='conv2/bn')
    conv2_relu      = tf.nn.relu(conv2_bn, name = 'conv2/relu')
    conv3_dw_pad    = tf.pad(conv2_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv3_dw        = convolution(conv3_dw_pad, group=128, strides=[1, 1], padding='VALID', name='conv3/dw')
    conv3_dw_bn     = batch_normalization(conv3_dw, variance_epsilon=9.999999747378752e-06, name='conv3/dw/bn')
    conv3_dw_relu   = tf.nn.relu(conv3_dw_bn, name = 'conv3/dw/relu')
    conv3           = convolution(conv3_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv3')
    conv3_bn        = batch_normalization(conv3, variance_epsilon=9.999999747378752e-06, name='conv3/bn')
    conv3_relu      = tf.nn.relu(conv3_bn, name = 'conv3/relu')
    conv4_dw_pad    = tf.pad(conv3_relu, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv4_dw        = convolution(conv4_dw_pad, group=128, strides=[2, 2], padding='VALID', name='conv4/dw')
    conv4_dw_bn     = batch_normalization(conv4_dw, variance_epsilon=9.999999747378752e-06, name='conv4/dw/bn')
    conv4_dw_relu   = tf.nn.relu(conv4_dw_bn, name = 'conv4/dw/relu')
    conv4           = convolution(conv4_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv4')
    conv4_bn        = batch_normalization(conv4, variance_epsilon=9.999999747378752e-06, name='conv4/bn')
    conv4_relu      = tf.nn.relu(conv4_bn, name = 'conv4/relu')
    conv5_dw_pad    = tf.pad(conv4_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_dw        = convolution(conv5_dw_pad, group=256, strides=[1, 1], padding='VALID', name='conv5/dw')
    conv5_dw_bn     = batch_normalization(conv5_dw, variance_epsilon=9.999999747378752e-06, name='conv5/dw/bn')
    conv5_dw_relu   = tf.nn.relu(conv5_dw_bn, name = 'conv5/dw/relu')
    conv5           = convolution(conv5_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv5')
    conv5_bn        = batch_normalization(conv5, variance_epsilon=9.999999747378752e-06, name='conv5/bn')
    conv5_relu      = tf.nn.relu(conv5_bn, name = 'conv5/relu')
    conv6_dw_pad    = tf.pad(conv5_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv6_dw        = convolution(conv6_dw_pad, group=256, strides=[2, 2], padding='VALID', name='conv6/dw')
    conv6_dw_bn     = batch_normalization(conv6_dw, variance_epsilon=9.999999747378752e-06, name='conv6/dw/bn')
    conv6_dw_relu   = tf.nn.relu(conv6_dw_bn, name = 'conv6/dw/relu')
    conv6           = convolution(conv6_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv6')
    conv6_bn        = batch_normalization(conv6, variance_epsilon=9.999999747378752e-06, name='conv6/bn')
    conv6_relu      = tf.nn.relu(conv6_bn, name = 'conv6/relu')
    conv7_dw_pad    = tf.pad(conv6_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv7_dw        = convolution(conv7_dw_pad, group=512, strides=[1, 1], padding='VALID', name='conv7/dw')
    conv7_dw_bn     = batch_normalization(conv7_dw, variance_epsilon=9.999999747378752e-06, name='conv7/dw/bn')
    conv7_dw_relu   = tf.nn.relu(conv7_dw_bn, name = 'conv7/dw/relu')
    conv7           = convolution(conv7_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv7')
    conv7_bn        = batch_normalization(conv7, variance_epsilon=9.999999747378752e-06, name='conv7/bn')
    conv7_relu      = tf.nn.relu(conv7_bn, name = 'conv7/relu')
    conv8_dw_pad    = tf.pad(conv7_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv8_dw        = convolution(conv8_dw_pad, group=512, strides=[1, 1], padding='VALID', name='conv8/dw')
    conv8_dw_bn     = batch_normalization(conv8_dw, variance_epsilon=9.999999747378752e-06, name='conv8/dw/bn')
    conv8_dw_relu   = tf.nn.relu(conv8_dw_bn, name = 'conv8/dw/relu')
    conv8           = convolution(conv8_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv8')
    conv8_bn        = batch_normalization(conv8, variance_epsilon=9.999999747378752e-06, name='conv8/bn')
    conv8_relu      = tf.nn.relu(conv8_bn, name = 'conv8/relu')
    conv9_dw_pad    = tf.pad(conv8_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv9_dw        = convolution(conv9_dw_pad, group=512, strides=[1, 1], padding='VALID', name='conv9/dw')
    conv9_dw_bn     = batch_normalization(conv9_dw, variance_epsilon=9.999999747378752e-06, name='conv9/dw/bn')
    conv9_dw_relu   = tf.nn.relu(conv9_dw_bn, name = 'conv9/dw/relu')
    conv9           = convolution(conv9_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv9')
    conv9_bn        = batch_normalization(conv9, variance_epsilon=9.999999747378752e-06, name='conv9/bn')
    conv9_relu      = tf.nn.relu(conv9_bn, name = 'conv9/relu')
    conv10_dw_pad   = tf.pad(conv9_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv10_dw       = convolution(conv10_dw_pad, group=512, strides=[1, 1], padding='VALID', name='conv10/dw')
    conv10_dw_bn    = batch_normalization(conv10_dw, variance_epsilon=9.999999747378752e-06, name='conv10/dw/bn')
    conv10_dw_relu  = tf.nn.relu(conv10_dw_bn, name = 'conv10/dw/relu')
    conv10          = convolution(conv10_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv10')
    conv10_bn       = batch_normalization(conv10, variance_epsilon=9.999999747378752e-06, name='conv10/bn')
    conv10_relu     = tf.nn.relu(conv10_bn, name = 'conv10/relu')
    conv11_dw_pad   = tf.pad(conv10_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv11_dw       = convolution(conv11_dw_pad, group=512, strides=[1, 1], padding='VALID', name='conv11/dw')
    conv11_dw_bn    = batch_normalization(conv11_dw, variance_epsilon=9.999999747378752e-06, name='conv11/dw/bn')
    conv11_dw_relu  = tf.nn.relu(conv11_dw_bn, name = 'conv11/dw/relu')
    conv11          = convolution(conv11_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv11')
    conv11_bn       = batch_normalization(conv11, variance_epsilon=9.999999747378752e-06, name='conv11/bn')
    conv11_relu     = tf.nn.relu(conv11_bn, name = 'conv11/relu')
    conv12_dw_pad   = tf.pad(conv11_relu, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv12_dw       = convolution(conv12_dw_pad, group=512, strides=[2, 2], padding='VALID', name='conv12/dw')
    conv11_mbox_loc = convolution(conv11_relu, group=1, strides=[1, 1], padding='VALID', name='conv11_mbox_loc')
    conv11_mbox_conf = convolution(conv11_relu, group=1, strides=[1, 1], padding='VALID', name='conv11_mbox_conf')
    conv12_dw_bn    = batch_normalization(conv12_dw, variance_epsilon=9.999999747378752e-06, name='conv12/dw/bn')
    conv12_dw_relu  = tf.nn.relu(conv12_dw_bn, name = 'conv12/dw/relu')
    conv12          = convolution(conv12_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv12')
    conv12_bn       = batch_normalization(conv12, variance_epsilon=9.999999747378752e-06, name='conv12/bn')
    conv12_relu     = tf.nn.relu(conv12_bn, name = 'conv12/relu')
    conv13_dw_pad   = tf.pad(conv12_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv13_dw       = convolution(conv13_dw_pad, group=1024, strides=[1, 1], padding='VALID', name='conv13/dw')
    conv13_dw_bn    = batch_normalization(conv13_dw, variance_epsilon=9.999999747378752e-06, name='conv13/dw/bn')
    conv13_dw_relu  = tf.nn.relu(conv13_dw_bn, name = 'conv13/dw/relu')
    conv13          = convolution(conv13_dw_relu, group=1, strides=[1, 1], padding='VALID', name='conv13')
    conv13_bn       = batch_normalization(conv13, variance_epsilon=9.999999747378752e-06, name='conv13/bn')
    conv13_relu     = tf.nn.relu(conv13_bn, name = 'conv13/relu')
    conv14_1        = convolution(conv13_relu, group=1, strides=[1, 1], padding='VALID', name='conv14_1')
    conv13_mbox_loc = convolution(conv13_relu, group=1, strides=[1, 1], padding='VALID', name='conv13_mbox_loc')
    conv13_mbox_conf = convolution(conv13_relu, group=1, strides=[1, 1], padding='VALID', name='conv13_mbox_conf')
    conv14_1_bn     = batch_normalization(conv14_1, variance_epsilon=9.999999747378752e-06, name='conv14_1/bn')
    conv14_1_relu   = tf.nn.relu(conv14_1_bn, name = 'conv14_1/relu')
    conv14_2_pad    = tf.pad(conv14_1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv14_2        = convolution(conv14_2_pad, group=1, strides=[2, 2], padding='VALID', name='conv14_2')
    conv14_2_bn     = batch_normalization(conv14_2, variance_epsilon=9.999999747378752e-06, name='conv14_2/bn')
    conv14_2_relu   = tf.nn.relu(conv14_2_bn, name = 'conv14_2/relu')
    conv15_1        = convolution(conv14_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv15_1')
    conv14_2_mbox_loc = convolution(conv14_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv14_2_mbox_loc')
    conv14_2_mbox_conf = convolution(conv14_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv14_2_mbox_conf')
    conv15_1_bn     = batch_normalization(conv15_1, variance_epsilon=9.999999747378752e-06, name='conv15_1/bn')
    conv15_1_relu   = tf.nn.relu(conv15_1_bn, name = 'conv15_1/relu')
    conv15_2_pad    = tf.pad(conv15_1_relu, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv15_2        = convolution(conv15_2_pad, group=1, strides=[2, 2], padding='VALID', name='conv15_2')
    conv15_2_bn     = batch_normalization(conv15_2, variance_epsilon=9.999999747378752e-06, name='conv15_2/bn')
    conv15_2_relu   = tf.nn.relu(conv15_2_bn, name = 'conv15_2/relu')
    conv16_1        = convolution(conv15_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv16_1')
    conv15_2_mbox_loc = convolution(conv15_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv15_2_mbox_loc')
    conv15_2_mbox_conf = convolution(conv15_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv15_2_mbox_conf')
    conv16_1_bn     = batch_normalization(conv16_1, variance_epsilon=9.999999747378752e-06, name='conv16_1/bn')
    conv16_1_relu   = tf.nn.relu(conv16_1_bn, name = 'conv16_1/relu')
    conv16_2_pad    = tf.pad(conv16_1_relu, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv16_2        = convolution(conv16_2_pad, group=1, strides=[2, 2], padding='VALID', name='conv16_2')
    conv16_2_bn     = batch_normalization(conv16_2, variance_epsilon=9.999999747378752e-06, name='conv16_2/bn')
    conv16_2_relu   = tf.nn.relu(conv16_2_bn, name = 'conv16_2/relu')
    conv17_1        = convolution(conv16_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv17_1')
    conv16_2_mbox_loc = convolution(conv16_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv16_2_mbox_loc')
    conv16_2_mbox_conf = convolution(conv16_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv16_2_mbox_conf')
    conv17_1_bn     = batch_normalization(conv17_1, variance_epsilon=9.999999747378752e-06, name='conv17_1/bn')
    conv17_1_relu   = tf.nn.relu(conv17_1_bn, name = 'conv17_1/relu')
    conv17_2_pad    = tf.pad(conv17_1_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv17_2        = convolution(conv17_2_pad, group=1, strides=[2, 2], padding='VALID', name='conv17_2')
    conv17_2_bn     = batch_normalization(conv17_2, variance_epsilon=9.999999747378752e-06, name='conv17_2/bn')
    conv17_2_relu   = tf.nn.relu(conv17_2_bn, name = 'conv17_2/relu')
    conv17_2_mbox_loc = convolution(conv17_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv17_2_mbox_loc')
    conv17_2_mbox_conf = convolution(conv17_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv17_2_mbox_conf')


    loc_flat_1 = tf.layers.Flatten()(conv11_mbox_loc)
    loc_flat_2 = tf.layers.Flatten()(conv13_mbox_loc)
    loc_flat_3 = tf.layers.Flatten()(conv14_2_mbox_loc)
    loc_flat_4 = tf.layers.Flatten()(conv15_2_mbox_loc) 
    loc_flat_5 = tf.layers.Flatten()(conv16_2_mbox_loc) 
    loc_flat_6 = tf.layers.Flatten()(conv17_2_mbox_loc)

    conf_flat_1 = tf.layers.Flatten()(conv11_mbox_conf)
    conf_flat_2 = tf.layers.Flatten()(conv13_mbox_conf)
    conf_flat_3 = tf.layers.Flatten()(conv14_2_mbox_conf)
    conf_flat_4 = tf.layers.Flatten()(conv15_2_mbox_conf) 
    conf_flat_5 = tf.layers.Flatten()(conv16_2_mbox_conf) 
    conf_flat_6 = tf.layers.Flatten()(conv17_2_mbox_conf)
    loc_concat=tf.concat([loc_flat_1,loc_flat_2,loc_flat_3,loc_flat_4,loc_flat_5,loc_flat_6],1)
    conf_concat=tf.concat([conf_flat_1,conf_flat_2,conf_flat_3,conf_flat_4,conf_flat_5,conf_flat_6],1)
    conf_concat=tf.layers.Flatten()(tf.nn.softmax(tf.reshape(conf_concat,[-1,21])))
    loc_concat=tf.layers.Flatten()(loc_concat)
    
    return data,loc_concat,conf_concat


    # return data, conv11_mbox_loc, conv11_mbox_conf, conv13_mbox_loc, conv13_mbox_conf, conv14_2_mbox_loc, conv14_2_mbox_conf, conv15_2_mbox_loc, conv15_2_mbox_conf, conv16_2_mbox_loc, conv16_2_mbox_conf, conv17_2_mbox_loc, conv17_2_mbox_conf


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, name=name, **kwargs)
    
    elif group == input.shape[-1].value:
        if len(kwargs['strides'])==2:
            kwargs['strides']=[1]+kwargs['strides']+[1]
            layer=  tf.nn.depthwise_conv2d(input,tf.expand_dims(tf.squeeze(w),-1), name=name, **kwargs)

    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, name=name, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)


    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer

def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)


