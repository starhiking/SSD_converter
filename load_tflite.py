import numpy as np
import tensorflow as tf
import cv2

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="ssd.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = (np.ones((300,300,3))*127).astype(np.float32).reshape(1,300,300,3)
index = input_details[0]['index']
interpreter.set_tensor(index, input_data)
interpreter.invoke()

conv17_2_mbox_loc = interpreter.get_tensor(output_details[2]['index']).reshape(-1)
print(conv17_2_mbox_loc)

conv17_2_mbox_conf = interpreter.get_tensor(output_details[3]['index']).reshape(-1)
print(conv17_2_mbox_conf)

