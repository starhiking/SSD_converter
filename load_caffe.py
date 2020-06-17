import caffe
import os
import cv2
import numpy as np
import time

caffe.set_mode_cpu()

caffemodel_path = "mv2_0.5_ssd/MobileNetSSD_nobn_100000.caffemodel"
net_txt_path = "mv2_0.5_ssd/MobileNetSSD_deploy_05_tf.prototxt"

caffe_model = caffe.Net(net_txt_path,caffemodel_path,caffe.TEST)

# transform_img = (np.ones((288,288,3))*127).astype(np.float32).reshape(1,3,288,288)
# transform_img = np.load('input.npy').transpose((2,0,1)).reshape(1,3,288,288)
transform_img = np.load('input_288.npy').transpose((2,0,1)).reshape(1,3,288,288)


caffe_model.blobs['data'].data[...] = transform_img

result = caffe_model.forward()

print(result['conv11_mbox_loc'].transpose((0,2,3,1)).reshape(-1)[0:100])
# print(result['conv11_mbox_conf'].transpose((0,2,3,1))
