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
    conv1_pad       = tf.pad(data, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv1           = convolution(conv1_pad, group=1, strides=[2, 2], padding='VALID', name='conv1')
    conv1_bn        = batch_normalization(conv1, variance_epsilon=9.999999747378752e-06, name='conv1/bn')
    relu1           = tf.nn.relu(conv1_bn, name = 'relu1')
    conv2_1_dw_pad  = tf.pad(relu1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv2_1_dw      = convolution(conv2_1_dw_pad, group=16, strides=[1, 1], padding='VALID', name='conv2_1/dw')
    conv2_1_dw_bn   = batch_normalization(conv2_1_dw, variance_epsilon=9.999999747378752e-06, name='conv2_1/dw/bn')
    relu2_1_dw      = tf.nn.relu(conv2_1_dw_bn, name = 'relu2_1/dw')
    conv2_1_sep     = convolution(relu2_1_dw, group=1, strides=[1, 1], padding='VALID', name='conv2_1/sep')
    conv2_1_sep_bn  = batch_normalization(conv2_1_sep, variance_epsilon=9.999999747378752e-06, name='conv2_1/sep/bn')
    relu2_1_sep     = tf.nn.relu(conv2_1_sep_bn, name = 'relu2_1/sep')
    conv2_2_dw_pad  = tf.pad(relu2_1_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv2_2_dw      = convolution(conv2_2_dw_pad, group=32, strides=[2, 2], padding='VALID', name='conv2_2/dw')
    conv2_2_dw_bn   = batch_normalization(conv2_2_dw, variance_epsilon=9.999999747378752e-06, name='conv2_2/dw/bn')
    relu2_2_dw      = tf.nn.relu(conv2_2_dw_bn, name = 'relu2_2/dw')
    conv2_2_sep     = convolution(relu2_2_dw, group=1, strides=[1, 1], padding='VALID', name='conv2_2/sep')
    conv2_2_sep_bn  = batch_normalization(conv2_2_sep, variance_epsilon=9.999999747378752e-06, name='conv2_2/sep/bn')
    relu2_2_sep     = tf.nn.relu(conv2_2_sep_bn, name = 'relu2_2/sep')
    conv3_1_dw_pad  = tf.pad(relu2_2_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv3_1_dw      = convolution(conv3_1_dw_pad, group=64, strides=[1, 1], padding='VALID', name='conv3_1/dw')
    conv3_1_dw_bn   = batch_normalization(conv3_1_dw, variance_epsilon=9.999999747378752e-06, name='conv3_1/dw/bn')
    relu3_1_dw      = tf.nn.relu(conv3_1_dw_bn, name = 'relu3_1/dw')
    conv3_1_sep     = convolution(relu3_1_dw, group=1, strides=[1, 1], padding='VALID', name='conv3_1/sep')
    conv3_1_sep_bn  = batch_normalization(conv3_1_sep, variance_epsilon=9.999999747378752e-06, name='conv3_1/sep/bn')
    relu3_1_sep     = tf.nn.relu(conv3_1_sep_bn, name = 'relu3_1/sep')
    conv3_2_dw_pad  = tf.pad(relu3_1_sep, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv3_2_dw      = convolution(conv3_2_dw_pad, group=64, strides=[2, 2], padding='VALID', name='conv3_2/dw')
    conv3_2_dw_bn   = batch_normalization(conv3_2_dw, variance_epsilon=9.999999747378752e-06, name='conv3_2/dw/bn')
    relu3_2_dw      = tf.nn.relu(conv3_2_dw_bn, name = 'relu3_2/dw')
    conv3_2_sep     = convolution(relu3_2_dw, group=1, strides=[1, 1], padding='VALID', name='conv3_2/sep')
    conv3_2_sep_bn  = batch_normalization(conv3_2_sep, variance_epsilon=9.999999747378752e-06, name='conv3_2/sep/bn')
    relu3_2_sep     = tf.nn.relu(conv3_2_sep_bn, name = 'relu3_2/sep')
    conv4_1_dw_pad  = tf.pad(relu3_2_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_1_dw      = convolution(conv4_1_dw_pad, group=128, strides=[1, 1], padding='VALID', name='conv4_1/dw')
    conv4_1_dw_bn   = batch_normalization(conv4_1_dw, variance_epsilon=9.999999747378752e-06, name='conv4_1/dw/bn')
    relu4_1_dw      = tf.nn.relu(conv4_1_dw_bn, name = 'relu4_1/dw')
    conv4_1_sep     = convolution(relu4_1_dw, group=1, strides=[1, 1], padding='VALID', name='conv4_1/sep')
    conv4_1_sep_bn  = batch_normalization(conv4_1_sep, variance_epsilon=9.999999747378752e-06, name='conv4_1/sep/bn')
    relu4_1_sep     = tf.nn.relu(conv4_1_sep_bn, name = 'relu4_1/sep')
    conv4_2_dw_pad  = tf.pad(relu4_1_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_2_dw      = convolution(conv4_2_dw_pad, group=128, strides=[2, 2], padding='VALID', name='conv4_2/dw')
    conv4_2_dw_bn   = batch_normalization(conv4_2_dw, variance_epsilon=9.999999747378752e-06, name='conv4_2/dw/bn')
    relu4_2_dw      = tf.nn.relu(conv4_2_dw_bn, name = 'relu4_2/dw')
    conv4_2_sep     = convolution(relu4_2_dw, group=1, strides=[1, 1], padding='VALID', name='conv4_2/sep')
    conv4_2_sep_bn  = batch_normalization(conv4_2_sep, variance_epsilon=9.999999747378752e-06, name='conv4_2/sep/bn')
    relu4_2_sep     = tf.nn.relu(conv4_2_sep_bn, name = 'relu4_2/sep')
    conv5_1_dw_pad  = tf.pad(relu4_2_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_1_dw      = convolution(conv5_1_dw_pad, group=256, strides=[1, 1], padding='VALID', name='conv5_1/dw')
    conv5_1_dw_bn   = batch_normalization(conv5_1_dw, variance_epsilon=9.999999747378752e-06, name='conv5_1/dw/bn')
    relu5_1_dw      = tf.nn.relu(conv5_1_dw_bn, name = 'relu5_1/dw')
    conv5_1_sep     = convolution(relu5_1_dw, group=1, strides=[1, 1], padding='VALID', name='conv5_1/sep')
    conv5_1_sep_bn  = batch_normalization(conv5_1_sep, variance_epsilon=9.999999747378752e-06, name='conv5_1/sep/bn')
    relu5_1_sep     = tf.nn.relu(conv5_1_sep_bn, name = 'relu5_1/sep')
    conv5_2_dw_pad  = tf.pad(relu5_1_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_2_dw      = convolution(conv5_2_dw_pad, group=256, strides=[1, 1], padding='VALID', name='conv5_2/dw')
    conv5_2_dw_bn   = batch_normalization(conv5_2_dw, variance_epsilon=9.999999747378752e-06, name='conv5_2/dw/bn')
    relu5_2_dw      = tf.nn.relu(conv5_2_dw_bn, name = 'relu5_2/dw')
    conv5_2_sep     = convolution(relu5_2_dw, group=1, strides=[1, 1], padding='VALID', name='conv5_2/sep')
    conv5_2_sep_bn  = batch_normalization(conv5_2_sep, variance_epsilon=9.999999747378752e-06, name='conv5_2/sep/bn')
    relu5_2_sep     = tf.nn.relu(conv5_2_sep_bn, name = 'relu5_2/sep')
    conv5_3_dw_pad  = tf.pad(relu5_2_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_3_dw      = convolution(conv5_3_dw_pad, group=256, strides=[1, 1], padding='VALID', name='conv5_3/dw')
    conv5_3_dw_bn   = batch_normalization(conv5_3_dw, variance_epsilon=9.999999747378752e-06, name='conv5_3/dw/bn')
    relu5_3_dw      = tf.nn.relu(conv5_3_dw_bn, name = 'relu5_3/dw')
    conv5_3_sep     = convolution(relu5_3_dw, group=1, strides=[1, 1], padding='VALID', name='conv5_3/sep')
    conv5_3_sep_bn  = batch_normalization(conv5_3_sep, variance_epsilon=9.999999747378752e-06, name='conv5_3/sep/bn')
    relu5_3_sep     = tf.nn.relu(conv5_3_sep_bn, name = 'relu5_3/sep')
    conv5_4_dw_pad  = tf.pad(relu5_3_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_4_dw      = convolution(conv5_4_dw_pad, group=256, strides=[1, 1], padding='VALID', name='conv5_4/dw')
    conv5_4_dw_bn   = batch_normalization(conv5_4_dw, variance_epsilon=9.999999747378752e-06, name='conv5_4/dw/bn')
    relu5_4_dw      = tf.nn.relu(conv5_4_dw_bn, name = 'relu5_4/dw')
    conv5_4_sep     = convolution(relu5_4_dw, group=1, strides=[1, 1], padding='VALID', name='conv5_4/sep')
    conv5_4_sep_bn  = batch_normalization(conv5_4_sep, variance_epsilon=9.999999747378752e-06, name='conv5_4/sep/bn')
    relu5_4_sep     = tf.nn.relu(conv5_4_sep_bn, name = 'relu5_4/sep')
    conv5_5_dw_pad  = tf.pad(relu5_4_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_5_dw      = convolution(conv5_5_dw_pad, group=256, strides=[1, 1], padding='VALID', name='conv5_5/dw')
    conv5_5_dw_bn   = batch_normalization(conv5_5_dw, variance_epsilon=9.999999747378752e-06, name='conv5_5/dw/bn')
    relu5_5_dw      = tf.nn.relu(conv5_5_dw_bn, name = 'relu5_5/dw')
    conv5_5_sep     = convolution(relu5_5_dw, group=1, strides=[1, 1], padding='VALID', name='conv5_5/sep')
    conv5_5_sep_bn  = batch_normalization(conv5_5_sep, variance_epsilon=9.999999747378752e-06, name='conv5_5/sep/bn')
    relu5_5_sep     = tf.nn.relu(conv5_5_sep_bn, name = 'relu5_5/sep')
    conv5_6_dw_pad  = tf.pad(relu5_5_sep, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv5_6_dw      = convolution(conv5_6_dw_pad, group=256, strides=[2, 2], padding='VALID', name='conv5_6/dw')
    conv11_mbox_loc = convolution(relu5_5_sep, group=1, strides=[1, 1], padding='VALID', name='conv11_mbox_loc')
    conv11_mbox_conf = convolution(relu5_5_sep, group=1, strides=[1, 1], padding='VALID', name='conv11_mbox_conf')
    conv5_6_dw_bn   = batch_normalization(conv5_6_dw, variance_epsilon=9.999999747378752e-06, name='conv5_6/dw/bn')
    relu5_6_dw      = tf.nn.relu(conv5_6_dw_bn, name = 'relu5_6/dw')
    conv5_6_sep     = convolution(relu5_6_dw, group=1, strides=[1, 1], padding='VALID', name='conv5_6/sep')
    conv5_6_sep_bn  = batch_normalization(conv5_6_sep, variance_epsilon=9.999999747378752e-06, name='conv5_6/sep/bn')
    relu5_6_sep     = tf.nn.relu(conv5_6_sep_bn, name = 'relu5_6/sep')
    conv6_dw_pad    = tf.pad(relu5_6_sep, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv6_dw        = convolution(conv6_dw_pad, group=512, strides=[1, 1], padding='VALID', name='conv6/dw')
    conv6_dw_bn     = batch_normalization(conv6_dw, variance_epsilon=9.999999747378752e-06, name='conv6/dw/bn')
    relu6_dw        = tf.nn.relu(conv6_dw_bn, name = 'relu6/dw')
    conv6_sep       = convolution(relu6_dw, group=1, strides=[1, 1], padding='VALID', name='conv6/sep')
    conv6_sep_bn    = batch_normalization(conv6_sep, variance_epsilon=9.999999747378752e-06, name='conv6/sep/bn')
    relu6_sep       = tf.nn.relu(conv6_sep_bn, name = 'relu6/sep')
    conv14_1        = convolution(relu6_sep, group=1, strides=[1, 1], padding='VALID', name='conv14_1')
    conv13_mbox_loc = convolution(relu6_sep, group=1, strides=[1, 1], padding='VALID', name='conv13_mbox_loc')
    conv13_mbox_conf = convolution(relu6_sep, group=1, strides=[1, 1], padding='VALID', name='conv13_mbox_conf')
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
    conf_concat=tf.layers.Flatten()(tf.nn.softmax(tf.reshape(conf_concat,[-1,4])))
    loc_concat=tf.layers.Flatten()(loc_concat)
    
    return data,loc_concat,conf_concat
    #return data,tf.concat([loc_concat,conf_concat],1)
    # return data, conv11_mbox_loc, conv11_mbox_conf, conv13_mbox_loc, conv13_mbox_conf, conv14_2_mbox_loc, conv14_2_mbox_conf, conv15_2_mbox_loc, conv15_2_mbox_conf, conv16_2_mbox_loc, conv16_2_mbox_conf, conv17_2_mbox_loc, conv17_2_mbox_conf


def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, name=name, **kwargs)
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

