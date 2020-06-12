
import argparse
import numpy as np
import sys
import os
import tensorflow as tf
from mmdnn.conversion.examples.imagenet_test import TestKit

class TestTF(TestKit):

    def __init__(self):
        super(TestTF, self).__init__()

        self.truth['mxnet']['resnet152-11k'] = [(1278, 0.49070787), (1277, 0.21392652), (282, 0.12979421), (1282, 0.066355646), (1224, 0.022040566)]
        self.outputs = {}
        self.input, self.output_0, self.output_1 = self.MainModel.KitModel(self.args.w)
        # self.input, self.model = self.MainModel.KitModel(self.args.w)
        # self.input, self.model, self.testop = self.MainModel.KitModel(self.args.w)


    def preprocess(self, image_path):
        x = super(TestTF, self).preprocess(image_path)
        self.data = np.expand_dims(x, 0)


    def print_result(self):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            predict = sess.run({self.output_0,self.output_1}, feed_dict = {self.input : self.data})

        super(TestTF, self).print_result(predict)


    def print_intermediate_result(self, layer_name, if_transpose = False):
        # testop = tf.get_default_graph().get_operation_by_name(layer_name)
        testop = self.testop
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            intermediate_output = sess.run(testop, feed_dict = {self.input : self.data})

        super(TestTF, self).print_intermediate_result(intermediate_output, if_transpose)


    def inference(self, image_path):
        self.preprocess(image_path)

        # self.print_intermediate_result('conv1_7x7_s2_1', True)

        self.print_result()

        self.test_truth()


    def dump(self, path = None):
        dump_tag = self.args.dump_tag
        if dump_tag == 'SERVING':
            tag_list = [tf.saved_model.tag_constants.SERVING]
        else:
            tag_list = [tf.saved_model.tag_constants.TRAINING]

        if path is None: path = self.args.dump
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            builder = tf.saved_model.builder.SavedModelBuilder(path)

            tensor_info_input = tf.saved_model.utils.build_tensor_info(self.input)
            tensor_info_output_0 = tf.saved_model.utils.build_tensor_info(self.output_0)
            tensor_info_output_1 = tf.saved_model.utils.build_tensor_info(self.output_1)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': tensor_info_input},
                    outputs={'output_0': tensor_info_output_0,"output_1":tensor_info_output_1},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

            builder.add_meta_graph_and_variables(
                sess,
                tag_list,
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                }
            )

            save_path = builder.save()

            print ('Tensorflow file is saved as [{}], generated by [{}.py] and [{}].'.format(
                save_path, self.args.n, self.args.w))


if __name__=='__main__':
    tester = TestTF()
    if tester.args.dump:
        if tester.args.dump_tag:
            tester.dump()
        else:
            raise ValueError("Need to provide the model type of Tensorflow model.")
    else:
        tester.inference(tester.args.image)