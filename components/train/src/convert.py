import tensorflow as tf
import os

from __future__ import print_function

import os
import shutil
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

import keras
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Dense, Conv2D
from keras.layers import BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import tensorflow.contrib.tensorrt as trt
# keras.mixed_precision.experimental.set_policy("default_mixed")
config = tf.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth = True

CONT_TRTIS_RESOURCE_DIR = 'trtis_resource'
tf_model_path = "/mnt/workspace/saved_model/tf_saved_model"
model_dir = "/mnt/workspace/saved_model/tsrt"
batch_size = 128
graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=config):
        # Create a TensorRT inference graph from a SavedModel:
        trt_graph = trt.create_inference_graph(
            input_graph_def=None,
            outputs=None,
            input_saved_model_dir=tf_model_path,
            input_saved_model_tags=[tag_constants.SERVING],
            max_batch_size=batch_size,
            max_workspace_size_bytes=2 << 30,
            precision_mode='fp16')

        print([n.name + '=>' + n.op for n in trt_graph.node])

        tf.io.write_graph(
            trt_graph,
            os.path.join(model_dir, "1"),
            'model.graphdef',
            as_text=False
        )
