import numpy as np
import tensorflow as tf
import cv2

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mv2_0.5_ssd.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# input_data = (np.ones((300,300,3))*127).astype(np.float32).reshape(1,300,300,3)
# input_data = np.load('input.npy').reshape(1,300,300,3)
input_data = np.load('input_288.npy').reshape(1,288,288,3)


index = input_details[0]['index']
interpreter.set_tensor(index, input_data)
interpreter.invoke()

conv17_2_mbox_loc = interpreter.get_tensor(output_details[0]['index'])
print(conv17_2_mbox_loc.reshape(-1)[0:100])

conv17_2_mbox_conf = interpreter.get_tensor(output_details[1]['index'])
print(conv17_2_mbox_conf.reshape(-1)[0:100])

