import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#RUN inference

mdir = "./createdModel/add_simple/"

interpreter = tf.lite.Interpreter(mdir+'add_simple.tflite',experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print('INPUT DETAILS:\n',input_details)
# print('OUTPUT DETAILS:\n',output_details)

# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape)*256 -128, dtype=np.int8)

# # input_data = np.array([[1,7],[35,66],[2,3],[56,11]],dtype=np.int8)
input_data1 = np.array([[1,7],[2,5]],dtype=np.int8)
input_data2 = np.array([[2,3],[4,9]],dtype=np.int8)
# # input_data = np.array([[1,7],[35,66]],dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data1)
interpreter.set_tensor(input_details[1]['index'], input_data2)
interpreter.get_tensor_details()
interpreter.invoke()

# print(interpreter.get_tensor_details())
print("Input1 Tensor: \n" + str(interpreter.get_tensor(0)))
print("Input2 Tensor: \n" + str(interpreter.get_tensor(1)))
print("Output Tensor: \n" + str(interpreter.get_tensor(2)))

golden_res= input_data1+input_data2
print("Golden Result:\n", golden_res)