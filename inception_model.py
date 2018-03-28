
# coding: utf-8

# In[ ]:


import numpy as np
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import h5py 
import os

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import graph_util

from scipy.misc import imread, imsave, imresize
import scipy.ndimage
import matplotlib
matplotlib.use('TkAgg') # choose appropriate rendering backend
from matplotlib import pyplot as plt



K.set_learning_phase(0)

model = applications.InceptionV3(include_top=False, weights='imagenet',input_shape=(299, 299, 3))

print model.summary()

SERVING="serve"
export_path = 'INCEPTION2'
#config = model.get_config()
#weights = model.get_weights()
#new_model = Sequential.from_config(config)
#new_model.set_weights(weights)


with tf.Graph().as_default() as g_input:
    input_b64 = tf.placeholder(shape=(1,),
                                dtype=tf.string,
                                name='input')
    input_bytes = tf.decode_base64(input_b64[0])
    image = tf.image.decode_image(input_bytes)
    image_f = tf.image.convert_image_dtype(image, dtype=tf.float32)
    input_image = tf.expand_dims(image_f, 0)
    output = tf.identity(input_image, name='input_image')

    # Convert to GraphDef
g_input_def = g_input.as_graph_def()
            

    
sess = K.get_session()

from tensorflow.python.framework import graph_util

    # Make GraphDef of Transfer Model
g_trans = sess.graph
g_trans_def = graph_util.convert_variables_to_constants(sess, 
                    g_trans.as_graph_def(),
                    #[model.output.name])
                      [model.output.name.replace(':0','')])
        
print('model.output.name is:')
print(model.output.name)
        
print('\n\n model.output.name.replace is:')
print(model.output.name.replace(':0',''))
    
with tf.Graph().as_default() as g_combined:
    x = tf.placeholder(tf.string, name="input_b64")

    im, = tf.import_graph_def(g_input_def,
                            input_map={'input:0': x},    
                            return_elements=["input_image:0"])

    pred, = tf.import_graph_def(g_trans_def,
            input_map={model.input.name: im}, 
            #{'batch_normalization_1/keras_learning_phase:0': False},
            return_elements=[model.output.name])
                          
                 
    
init_op = tf.global_variables_initializer()
sess2=tf.Session()
sess2.run(init_op)
inputs = {"image_bytes": tf.saved_model.utils.build_tensor_info(x)}
outputs = {"outputs":tf.saved_model.utils.build_tensor_info(pred)}
signature =tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        # save as SavedModel
b = tf.saved_model.builder.SavedModelBuilder(export_path)
b.add_meta_graph_and_variables(sess2,
                    [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={'serving_default': signature})
        
b.save()

