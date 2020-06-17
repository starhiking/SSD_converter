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

    data            = tf.placeholder(tf.float32, shape = (None, 288, 288, 3), name = 'data')
    conv1_pad       = tf.pad(data, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv1           = convolution(conv1_pad, group=1, strides=[2, 2], padding='VALID', name='conv1')
    relu1           = tf.nn.relu(conv1, name = 'relu1')
    conv2_1_expand  = convolution(relu1, group=1, strides=[1, 1], padding='VALID', name='conv2_1/expand')
    relu2_1_expand  = tf.nn.relu(conv2_1_expand, name = 'relu2_1/expand')
    conv2_1_dwise_pad = tf.pad(relu2_1_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv2_1_dwise   = convolution(conv2_1_dwise_pad, group=16, strides=[1, 1], padding='VALID', name='conv2_1/dwise')
    relu2_1_dwise   = tf.nn.relu(conv2_1_dwise, name = 'relu2_1/dwise')
    conv2_1_linear  = convolution(relu2_1_dwise, group=1, strides=[1, 1], padding='VALID', name='conv2_1/linear')
    conv2_2_expand  = convolution(conv2_1_linear, group=1, strides=[1, 1], padding='VALID', name='conv2_2/expand')
    relu2_2_expand  = tf.nn.relu(conv2_2_expand, name = 'relu2_2/expand')
    conv2_2_dwise_pad = tf.pad(relu2_2_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv2_2_dwise   = convolution(conv2_2_dwise_pad, group=48, strides=[2, 2], padding='VALID', name='conv2_2/dwise')
    relu2_2_dwise   = tf.nn.relu(conv2_2_dwise, name = 'relu2_2/dwise')
    conv2_2_linear  = convolution(relu2_2_dwise, group=1, strides=[1, 1], padding='VALID', name='conv2_2/linear')
    conv3_1_expand  = convolution(conv2_2_linear, group=1, strides=[1, 1], padding='VALID', name='conv3_1/expand')
    relu3_1_expand  = tf.nn.relu(conv3_1_expand, name = 'relu3_1/expand')
    conv3_1_dwise_pad = tf.pad(relu3_1_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv3_1_dwise   = convolution(conv3_1_dwise_pad, group=72, strides=[1, 1], padding='VALID', name='conv3_1/dwise')
    relu3_1_dwise   = tf.nn.relu(conv3_1_dwise, name = 'relu3_1/dwise')
    conv3_1_linear  = convolution(relu3_1_dwise, group=1, strides=[1, 1], padding='VALID', name='conv3_1/linear')
    block_3_1       = conv2_2_linear + conv3_1_linear
    conv3_2_expand  = convolution(block_3_1, group=1, strides=[1, 1], padding='VALID', name='conv3_2/expand')
    relu3_2_expand  = tf.nn.relu(conv3_2_expand, name = 'relu3_2/expand')
    conv3_2_dwise_pad = tf.pad(relu3_2_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv3_2_dwise   = convolution(conv3_2_dwise_pad, group=72, strides=[2, 2], padding='VALID', name='conv3_2/dwise')
    relu3_2_dwise   = tf.nn.relu(conv3_2_dwise, name = 'relu3_2/dwise')
    conv3_2_linear  = convolution(relu3_2_dwise, group=1, strides=[1, 1], padding='VALID', name='conv3_2/linear')
    conv4_1_expand  = convolution(conv3_2_linear, group=1, strides=[1, 1], padding='VALID', name='conv4_1/expand')
    relu4_1_expand  = tf.nn.relu(conv4_1_expand, name = 'relu4_1/expand')
    conv4_1_dwise_pad = tf.pad(relu4_1_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_1_dwise   = convolution(conv4_1_dwise_pad, group=96, strides=[1, 1], padding='VALID', name='conv4_1/dwise')
    relu4_1_dwise   = tf.nn.relu(conv4_1_dwise, name = 'relu4_1/dwise')
    conv4_1_linear  = convolution(relu4_1_dwise, group=1, strides=[1, 1], padding='VALID', name='conv4_1/linear')
    block_4_1       = conv3_2_linear + conv4_1_linear
    conv4_2_expand  = convolution(block_4_1, group=1, strides=[1, 1], padding='VALID', name='conv4_2/expand')
    relu4_2_expand  = tf.nn.relu(conv4_2_expand, name = 'relu4_2/expand')
    conv4_2_dwise_pad = tf.pad(relu4_2_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_2_dwise   = convolution(conv4_2_dwise_pad, group=96, strides=[1, 1], padding='VALID', name='conv4_2/dwise')
    relu4_2_dwise   = tf.nn.relu(conv4_2_dwise, name = 'relu4_2/dwise')
    conv4_2_linear  = convolution(relu4_2_dwise, group=1, strides=[1, 1], padding='VALID', name='conv4_2/linear')
    block_4_2       = block_4_1 + conv4_2_linear
    conv4_3_expand  = convolution(block_4_2, group=1, strides=[1, 1], padding='VALID', name='conv4_3/expand')
    relu4_3_expand  = tf.nn.relu(conv4_3_expand, name = 'relu4_3/expand')
    conv4_3_dwise_pad = tf.pad(relu4_3_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_3_dwise   = convolution(conv4_3_dwise_pad, group=96, strides=[1, 1], padding='VALID', name='conv4_3/dwise')
    relu4_3_dwise   = tf.nn.relu(conv4_3_dwise, name = 'relu4_3/dwise')
    conv4_3_linear  = convolution(relu4_3_dwise, group=1, strides=[1, 1], padding='VALID', name='conv4_3/linear')
    conv4_4_expand  = convolution(conv4_3_linear, group=1, strides=[1, 1], padding='VALID', name='conv4_4/expand')
    relu4_4_expand  = tf.nn.relu(conv4_4_expand, name = 'relu4_4/expand')
    conv4_4_dwise_pad = tf.pad(relu4_4_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_4_dwise   = convolution(conv4_4_dwise_pad, group=192, strides=[1, 1], padding='VALID', name='conv4_4/dwise')
    relu4_4_dwise   = tf.nn.relu(conv4_4_dwise, name = 'relu4_4/dwise')
    conv4_4_linear  = convolution(relu4_4_dwise, group=1, strides=[1, 1], padding='VALID', name='conv4_4/linear')
    block_4_4       = conv4_3_linear + conv4_4_linear
    conv4_5_expand  = convolution(block_4_4, group=1, strides=[1, 1], padding='VALID', name='conv4_5/expand')
    relu4_5_expand  = tf.nn.relu(conv4_5_expand, name = 'relu4_5/expand')
    conv4_5_dwise_pad = tf.pad(relu4_5_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_5_dwise   = convolution(conv4_5_dwise_pad, group=192, strides=[1, 1], padding='VALID', name='conv4_5/dwise')
    relu4_5_dwise   = tf.nn.relu(conv4_5_dwise, name = 'relu4_5/dwise')
    conv4_5_linear  = convolution(relu4_5_dwise, group=1, strides=[1, 1], padding='VALID', name='conv4_5/linear')
    block_4_5       = block_4_4 + conv4_5_linear
    conv4_6_expand  = convolution(block_4_5, group=1, strides=[1, 1], padding='VALID', name='conv4_6/expand')
    relu4_6_expand  = tf.nn.relu(conv4_6_expand, name = 'relu4_6/expand')
    conv4_6_dwise_pad = tf.pad(relu4_6_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_6_dwise   = convolution(conv4_6_dwise_pad, group=192, strides=[1, 1], padding='VALID', name='conv4_6/dwise')
    relu4_6_dwise   = tf.nn.relu(conv4_6_dwise, name = 'relu4_6/dwise')
    conv4_6_linear  = convolution(relu4_6_dwise, group=1, strides=[1, 1], padding='VALID', name='conv4_6/linear')
    block_4_6       = block_4_5 + conv4_6_linear
    conv4_7_expand  = convolution(block_4_6, group=1, strides=[1, 1], padding='VALID', name='conv4_7/expand')
    relu4_7_expand  = tf.nn.relu(conv4_7_expand, name = 'relu4_7/expand')
    conv4_7_dwise_pad = tf.pad(relu4_7_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_7_dwise   = convolution(conv4_7_dwise_pad, group=192, strides=[2, 2], padding='VALID', name='conv4_7/dwise')
    conv5_mbox_loc  = convolution(relu4_7_expand, group=1, strides=[1, 1], padding='VALID', name='conv5_mbox_loc')
    conv5_mbox_conf_new_new = convolution(relu4_7_expand, group=1, strides=[1, 1], padding='VALID', name='conv5_mbox_conf_new_new')
    relu4_7_dwise   = tf.nn.relu(conv4_7_dwise, name = 'relu4_7/dwise')
    conv4_7_linear  = convolution(relu4_7_dwise, group=1, strides=[1, 1], padding='VALID', name='conv4_7/linear')
    conv5_1_expand  = convolution(conv4_7_linear, group=1, strides=[1, 1], padding='VALID', name='conv5_1/expand')
    relu5_1_expand  = tf.nn.relu(conv5_1_expand, name = 'relu5_1/expand')
    conv5_1_dwise_pad = tf.pad(relu5_1_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_1_dwise   = convolution(conv5_1_dwise_pad, group=288, strides=[1, 1], padding='VALID', name='conv5_1/dwise')
    relu5_1_dwise   = tf.nn.relu(conv5_1_dwise, name = 'relu5_1/dwise')
    conv5_1_linear  = convolution(relu5_1_dwise, group=1, strides=[1, 1], padding='VALID', name='conv5_1/linear')
    block_5_1       = conv4_7_linear + conv5_1_linear
    conv5_2_expand  = convolution(block_5_1, group=1, strides=[1, 1], padding='VALID', name='conv5_2/expand')
    relu5_2_expand  = tf.nn.relu(conv5_2_expand, name = 'relu5_2/expand')
    conv5_2_dwise_pad = tf.pad(relu5_2_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_2_dwise   = convolution(conv5_2_dwise_pad, group=288, strides=[1, 1], padding='VALID', name='conv5_2/dwise')
    relu5_2_dwise   = tf.nn.relu(conv5_2_dwise, name = 'relu5_2/dwise')
    conv5_2_linear  = convolution(relu5_2_dwise, group=1, strides=[1, 1], padding='VALID', name='conv5_2/linear')
    block_5_2       = block_5_1 + conv5_2_linear
    conv5_3_expand  = convolution(block_5_2, group=1, strides=[1, 1], padding='VALID', name='conv5_3/expand')
    relu5_3_expand  = tf.nn.relu(conv5_3_expand, name = 'relu5_3/expand')
    conv5_3_dwise_pad = tf.pad(relu5_3_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_3_dwise   = convolution(conv5_3_dwise_pad, group=288, strides=[2, 2], padding='VALID', name='conv5_3/dwise')
    conv11_mbox_loc_new = convolution(relu5_3_expand, group=1, strides=[1, 1], padding='VALID', name='conv11_mbox_loc_new')
    conv11_mbox_conf_new_new_new = convolution(relu5_3_expand, group=1, strides=[1, 1], padding='VALID', name='conv11_mbox_conf_new_new_new')
    relu5_3_dwise   = tf.nn.relu(conv5_3_dwise, name = 'relu5_3/dwise')
    conv5_3_linear  = convolution(relu5_3_dwise, group=1, strides=[1, 1], padding='VALID', name='conv5_3/linear')
    conv6_1_expand  = convolution(conv5_3_linear, group=1, strides=[1, 1], padding='VALID', name='conv6_1/expand')
    relu6_1_expand  = tf.nn.relu(conv6_1_expand, name = 'relu6_1/expand')
    conv6_1_dwise_pad = tf.pad(relu6_1_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv6_1_dwise   = convolution(conv6_1_dwise_pad, group=480, strides=[1, 1], padding='VALID', name='conv6_1/dwise')
    relu6_1_dwise   = tf.nn.relu(conv6_1_dwise, name = 'relu6_1/dwise')
    conv6_1_linear  = convolution(relu6_1_dwise, group=1, strides=[1, 1], padding='VALID', name='conv6_1/linear')
    block_6_1       = conv5_3_linear + conv6_1_linear
    conv6_2_expand  = convolution(block_6_1, group=1, strides=[1, 1], padding='VALID', name='conv6_2/expand')
    relu6_2_expand  = tf.nn.relu(conv6_2_expand, name = 'relu6_2/expand')
    conv6_2_dwise_pad = tf.pad(relu6_2_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv6_2_dwise   = convolution(conv6_2_dwise_pad, group=480, strides=[1, 1], padding='VALID', name='conv6_2/dwise')
    relu6_2_dwise   = tf.nn.relu(conv6_2_dwise, name = 'relu6_2/dwise')
    conv6_2_linear  = convolution(relu6_2_dwise, group=1, strides=[1, 1], padding='VALID', name='conv6_2/linear')
    block_6_2       = block_6_1 + conv6_2_linear
    conv6_3_expand  = convolution(block_6_2, group=1, strides=[1, 1], padding='VALID', name='conv6_3/expand')
    relu6_3_expand  = tf.nn.relu(conv6_3_expand, name = 'relu6_3/expand')
    conv6_3_dwise_pad = tf.pad(relu6_3_expand, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv6_3_dwise   = convolution(conv6_3_dwise_pad, group=480, strides=[1, 1], padding='VALID', name='conv6_3/dwise')
    relu6_3_dwise   = tf.nn.relu(conv6_3_dwise, name = 'relu6_3/dwise')
    conv6_3_linear  = convolution(relu6_3_dwise, group=1, strides=[1, 1], padding='VALID', name='conv6_3/linear')
    conv6_4         = convolution(conv6_3_linear, group=1, strides=[1, 1], padding='VALID', name='conv6_4')
    relu6_4         = tf.nn.relu(conv6_4, name = 'relu6_4')
    conv14_1        = convolution(relu6_4, group=1, strides=[1, 1], padding='VALID', name='conv14_1')
    conv13_mbox_loc = convolution(relu6_4, group=1, strides=[1, 1], padding='VALID', name='conv13_mbox_loc')
    conv13_mbox_conf_new_new = convolution(relu6_4, group=1, strides=[1, 1], padding='VALID', name='conv13_mbox_conf_new_new')
    conv14_1_relu   = tf.nn.relu(conv14_1, name = 'conv14_1/relu')
    conv14_2_depthwise_pad = tf.pad(conv14_1_relu, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv14_2_depthwise = convolution(conv14_2_depthwise_pad, group=128, strides=[2, 2], padding='VALID', name='conv14_2/depthwise')
    conv14_2_depthwise_relu = tf.nn.relu(conv14_2_depthwise, name = 'conv14_2/depthwise/relu')
    conv14_2        = convolution(conv14_2_depthwise_relu, group=1, strides=[1, 1], padding='VALID', name='conv14_2')
    conv14_2_relu   = tf.nn.relu(conv14_2, name = 'conv14_2/relu')
    conv15_1        = convolution(conv14_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv15_1')
    conv14_2_mbox_loc = convolution(conv14_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv14_2_mbox_loc')
    conv14_2_mbox_conf_new_new = convolution(conv14_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv14_2_mbox_conf_new_new')
    conv15_1_relu   = tf.nn.relu(conv15_1, name = 'conv15_1/relu')
    conv15_2_depthwise_pad = tf.pad(conv15_1_relu, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv15_2_depthwise = convolution(conv15_2_depthwise_pad, group=64, strides=[2, 2], padding='VALID', name='conv15_2/depthwise')
    conv15_2_depthwise_relu = tf.nn.relu(conv15_2_depthwise, name = 'conv15_2/depthwise/relu')
    conv15_2        = convolution(conv15_2_depthwise_relu, group=1, strides=[1, 1], padding='VALID', name='conv15_2')
    conv15_2_relu   = tf.nn.relu(conv15_2, name = 'conv15_2/relu')
    conv16_1        = convolution(conv15_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv16_1')
    conv15_2_mbox_loc = convolution(conv15_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv15_2_mbox_loc')
    conv15_2_mbox_conf_new_new = convolution(conv15_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv15_2_mbox_conf_new_new')
    conv16_1_relu   = tf.nn.relu(conv16_1, name = 'conv16_1/relu')
    conv16_2_depthwise_pad = tf.pad(conv16_1_relu, paddings = [[0, 0], [1, 2], [1, 2], [0, 0]])
    conv16_2_depthwise = convolution(conv16_2_depthwise_pad, group=64, strides=[2, 2], padding='VALID', name='conv16_2/depthwise')
    conv16_2_depthwise_relu = tf.nn.relu(conv16_2_depthwise, name = 'conv16_2/depthwise/relu')
    conv16_2        = convolution(conv16_2_depthwise_relu, group=1, strides=[1, 1], padding='VALID', name='conv16_2')
    conv16_2_relu   = tf.nn.relu(conv16_2, name = 'conv16_2/relu')
    conv16_2_mbox_loc = convolution(conv16_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv16_2_mbox_loc')
    conv16_2_mbox_conf_new_new = convolution(conv16_2_relu, group=1, strides=[1, 1], padding='VALID', name='conv16_2_mbox_conf_new_new')
    

    loc_flat_1 = tf.layers.Flatten()(conv11_mbox_loc_new)
    loc_flat_2 = tf.layers.Flatten()(conv13_mbox_loc)
    loc_flat_3 = tf.layers.Flatten()(conv14_2_mbox_loc)
    loc_flat_4 = tf.layers.Flatten()(conv15_2_mbox_loc) 
    loc_flat_5 = tf.layers.Flatten()(conv16_2_mbox_loc) 

    conf_flat_1 = tf.layers.Flatten()(conv11_mbox_conf_new_new_new)
    conf_flat_2 = tf.layers.Flatten()(conv13_mbox_conf_new_new)
    conf_flat_3 = tf.layers.Flatten()(conv14_2_mbox_conf_new_new)
    conf_flat_4 = tf.layers.Flatten()(conv15_2_mbox_conf_new_new) 
    conf_flat_5 = tf.layers.Flatten()(conv16_2_mbox_conf_new_new) 
    loc_concat=tf.concat([loc_flat_1,loc_flat_2,loc_flat_3,loc_flat_4,loc_flat_5],1)
    conf_concat=tf.concat([conf_flat_1,conf_flat_2,conf_flat_3,conf_flat_4,conf_flat_5],1)
    conf_concat=tf.layers.Flatten()(tf.nn.softmax(tf.reshape(conf_concat,[-1,4])))
    loc_concat=tf.layers.Flatten()(loc_concat)
    
    return data,loc_concat,conf_concat
    
    # return data, conv5_mbox_loc, conv5_mbox_conf_new_new, conv11_mbox_loc_new, conv11_mbox_conf_new_new_new, conv13_mbox_loc, conv13_mbox_conf_new_new, conv14_2_mbox_loc, conv14_2_mbox_conf_new_new, conv15_2_mbox_loc, conv15_2_mbox_conf_new_new, conv16_2_mbox_loc, conv16_2_mbox_conf_new_new


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

