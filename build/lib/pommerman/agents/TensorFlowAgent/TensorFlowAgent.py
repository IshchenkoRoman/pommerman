from pommerman.agents import BaseAgent
from pommerman import characters

import tensorflow as tf
from tensorflow.python.client import device_lib


class TensorFlowAgent(BaseAgent):
    """TensorFlowAgent."""

    def __init__(self, character=characters.Bomber):
        super(TensorFlowAgent, self).__init__(character)
        self.model = self.buildModel()

    def blueprintModel(self):

        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, shape=[None, 11], name="input_board")

        W1 = tf.placeholder(tf.float32, shape=[1,1,11,11], name="W1")
        W2 = tf.placeholder(tf.float32, shape=[2,2,11,10], name="W2")

        T1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="SAME")
        T2 = tf.nn.conv2d(X, W2, strides=[1,1,1,1], padding="SAME")

        inception_1 = tf.tf.concat([T1, T2], 0  )
        inception_1_norm = tf.contrib.layers.batch_norm(inception_1)
        A1 = tf.nn.relu(inception_1_norm)

        flaten = tf.contrib.layers.flatten(A1)

        fc1 = tf.contrib.layers.fully_connected(flaten, num_outputs=121,weights_initializer = tf.contrib.layers.xavier_initializer(seed=0), \
                                                weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001), activation_fn=None)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=121,weights_initializer = tf.contrib.layers.xavier_initializer(seed=0), \
                                                weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001), activation_fn=None)

        fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs=5,weights_initializer = tf.contrib.layers.xavier_initializer(seed=0), \
                                                weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001), activation_fn=None)

        sf_max = tf.nn.softmax(fc3)

        return (sf_max)

    def buildModel(self):

        def get_available_gpus():
            local_device_protos = device_lib.list_local_devices()
            gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
            cpu_list = [x.name for x in local_device_protos if x.device_type == 'CPU']
            return (gpu_list, cpu_list)

        gpu_list, cpu_list = get_available_gpus()

        predict_nn = self.blueprintModel()

    def act(self, obs, action_space):

        return None

