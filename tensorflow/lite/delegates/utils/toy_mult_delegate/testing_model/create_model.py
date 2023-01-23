import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Shape Params

#l1
x = 2
y = 2
input_shape = (x,y)
mdir = "./createdModel/"
# print(mdir)
input1 = tf.keras.layers.Input(shape=(x,),batch_size=y)
input2 = tf.keras.layers.Input(shape=(x,),batch_size=y)

#simple add model

added = tf.keras.layers.Add(dtype='int32')([input1, input2])
add_simple = tf.keras.models.Model(inputs=[input1, input2], outputs=added)
add_simple.summary()
add_simple.compile(optimizer='sgd', loss='mse')
add_simple.fit([np.random.rand(x,y),np.random.rand(x,y)] , np.random.rand(x,y),batch_size=1, epochs=10)
mdir = "./createdModel/add_simple/"
add_simple.save(mdir)

#simple multiplication model
# mult = tf.keras.layers.Multiply()([input1, input2])
# mult_simple = tf.keras.models.Model(inputs=[input1, input2], outputs=mult)
# # mult_simple.summary()
# mult_simple.compile(optimizer='sgd', loss='mse')
# mult_simple.fit([np.random.rand(x,y),np.random.rand(x,y)] , np.random.rand(x,y),batch_size=1, epochs=10)
# mdir = "./createdModel/mult_simple/"
# mult_simple.save(mdir)

#
def representative_dataset():
    for _ in range(1):
        data = np.random.rand(x,y)
#         data = np.ones((depth,in_size),)
        yield [data.astype(np.float32),data.astype(np.float32)]
        # yield [data.astype(np.int32),data.astype(np.int32)]
        
converter = tf.lite.TFLiteConverter.from_saved_model(mdir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

with open(mdir+'add_simple.tflite', 'wb') as f:
    f.write(tflite_model)