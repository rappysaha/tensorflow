import numpy as np
import tensorflow as tf

a=10
print(a)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/root/Workspace/tensorflow/models/add_simple.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape1 = input_details[0]['shape']
input_data1 = np.array(np.random.random_sample(input_shape1), dtype=np.float32)
input_shape2 = input_details[1]['shape']
input_data2 = np.array(np.random.random_sample(input_shape2), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data1)
interpreter.set_tensor(input_details[1]['index'], input_data2)

# print(input_details[1]['index'])
print(output_details)
print('input data1=\n',input_data1)
print('input data2=\n',input_data2)

interpreter.invoke()


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

print('output data=\n',output_data)

test_output_data= input_data1 + input_data2

print('test output data=\n',test_output_data)

diff = test_output_data - output_data

print('diff=\n',diff)