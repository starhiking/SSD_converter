import caffe
import os
import cv2
import numpy as np
import time

caffe.set_mode_cpu()

caffemodel_path = "./mbv1_ssd_05_300_300_car_4_models3_iter_60000.caffemodel"
net_txt_path = "./MobileNetSSD_deploy_backbone.prototxt"

caffe_model = caffe.Net(net_txt_path,caffemodel_path,caffe.TEST)

# transform_img = (np.ones((300,300,3))*127).astype(np.float32).reshape(1,3,300,300)
transform_img = np.load('input.npy').transpose((2,0,1)).reshape(1,3,300,300)

caffe_model.blobs['data'].data[...] = transform_img

result = caffe_model.forward()

print(result['conv16_2_mbox_loc'].reshape(-1))
print(result['conv16_2_mbox_conf'].reshape(-1))

