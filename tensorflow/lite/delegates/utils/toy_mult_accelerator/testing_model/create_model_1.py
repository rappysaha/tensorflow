import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(2)
    self.dense2 = tf.keras.layers.Dense(2)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

class MyModel2(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(2)

  def call(self, inputs):
    return self.dense1(inputs)

x=224*220*3


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(x,)))
model.add(MyModel2())

model.summary()


mdir = "./createdModel/dense/"
model.save(mdir)

def representative_dataset():
    for _ in range(1):
        data = np.random.rand(x,)
#         data = np.ones((depth,in_size),)
        yield [data.astype(np.float32)]
        # yield [data.astype(np.int32),data.astype(np.int32)]
        
converter = tf.lite.TFLiteConverter.from_saved_model(mdir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

with open(mdir+'denseTest4.tflite', 'wb') as f:
    f.write(tflite_model)