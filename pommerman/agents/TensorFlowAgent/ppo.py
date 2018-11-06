import tensorflow as tf
import numpy as np
import os

import glob

class BasePPO(object):

    def __init__(self, action_space, observation_space, scope, args, path_models, type="CNN"):

        self.path_models = path_models

        self.action_space = action_space
        self.observation_space = observation_space

        self.scope = scope
        self._type = type

        self.action_bound = [0.001, 5.999]
        self.num_state = self.observation_space
        self.num_action = 1

        self.args = args
        self.cliprange = args.cliprange

        if not os.path.exists(self.path_models):
            os.makedirs(self.path_models)
        with tf.variable_scope(self._type + "input_observation"):
            # self.s = tf.placeholder("float", [None, self.num_state])
            # self.s = tf.placeholder("float", [None, 3, 11, 11], name="input_observation")
            if self._type == "CNN" or self._type == "Simple":
                self.s = tf.placeholder("float", [None, 11], name="state")
            else:
                self.s  = tf.placeholder("float", [None, 3, 11, 11], name="state")
        with tf.variable_scope(self._type + "input_ammos"):
            self.ammo = tf.placeholder(tf.int32, name="input_ammos")
        with tf.variable_scope(self._type + "action"):
            self.a = tf.placeholder(shape=[None, self.num_action], dtype=tf.float32)
        with tf.variable_scope(self._type + "target_value"):
            self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope(self._type + "advantages"):
            self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope(self._type + "alive_agents"):
            self.alive_agents = tf.placeholder(shape=[4], dtype=tf.int8)

    def build_critic_net(self, net):

        raise NotImplementedError("You cann't instantiate this class")

    def build_actor_net(self, scope, trainable):

        raise NotImplementedError("You cann't instantiate this class")

    def build_net(self):

        self.value = self.build_critic_net("value_net")

        if self._type == "CNN":
            pi, pi_param = self.build_actor_net_cnn("actor_net", trainable=True)
            old_pi, old_pi_param = self.build_actor_net_cnn("old_actor_net", trainable=False)
        elif self._type == "Simple":
            pi, pi_param = self.build_actor_net_simple("actor_net", trainable=True)
            old_pi, old_pi_param = self.build_actor_net_simple("old_actor_net", trainable=False)
        self.syn_old_pi = [oldp.assign(p) for p, oldp in zip(pi_param, old_pi_param)]
        self.sample_op = tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), self.action_bound[0], self.action_bound[1])[0]

        with tf.variable_scope('critic_loss'):
            self.adv = self.y - self.value
            self.critic_loss = tf.reduce_mean(tf.square(self.adv))

        with tf.variable_scope("actor_loss"):
            ratio = pi.prob(self.a) / old_pi.prob(self.a)
            pg_losses = self.advantage * ratio
            pg_losses2 = self.advantage * tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            self.actor_loss = -tf.reduce_mean(tf.minimum(pg_losses, pg_losses2))

    def load_model(self, sess, saver):

        checkpoint = tf.train.get_checkpoint_state(self.path_models)
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("...................Model restored to global..............")
        else:
            print("..................No model is found....................")

    def save_model(self, sess, saver, time_step):

        print("..................Save model....................")
        saver.save(sess, "".join([self.path_models, "/", "Pommerman", "-", str(time_step), ".ckpt"]))

    def choose_action(self, s, ammo, sess):

        a = sess.run(self.sample_op, {self.s: s, self.ammo: ammo})

        return (a)

    def get_v(self, s, sess):

        return (sess.run(self.value, {self.s: s})[0,0])

class MlpPPO(BasePPO):

    def __init__(self, action_space, observation_space, scope, args, path_models):

        super().__init__(action_space, observation_space, scope, args, path_models)
        self.build_net()

    def _cnn_block(self, scope, num_outputs=100, trainable=False):

        with tf.variable_scope(scope):

            board = tf.manip.reshape(self.s, [-1, 11, 11, 1], name="reshaped_input")

            filter_one = tf.get_variable("w01", [1, 1, 1, num_outputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)
            filter_two = tf.get_variable("w02", [2, 2, num_outputs, num_outputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)

            stage_one = tf.nn.conv2d(input=board, filter=filter_one, strides=[1, 1, 1, 1], padding="SAME", name="stage_one")
            stage_two = tf.nn.conv2d(input=stage_one, filter=filter_two, strides=[1, 1, 1, 1], padding="SAME", name="stage_two")
            stage_two_flatten = tf.contrib.layers.flatten(stage_two)

        return (stage_two_flatten)

    def build_critic_net(self, scope):

        with tf.variable_scope(scope):

            board = tf.manip.reshape(self.s, [-1, 121])

            dl1 = tf.contrib.layers.fully_connected(inputs=board, num_outputs=100, activation_fn=tf.nn.relu, scope='cnn_dl1')

            value = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=1, activation_fn=None, scope="cnn_value")

            return (value)

    def build_actor_net_cnn(self, scope, trainable):

        inception_flatten = self._cnn_block(scope, num_outputs=200, trainable=trainable)

        with tf.variable_scope(scope):

            dl1 = tf.contrib.layers.fully_connected(inputs=inception_flatten, num_outputs=200, activation_fn=tf.nn.relu, trainable=trainable, scope="dl1")

            dl1 = tf.clip_by_value(dl1, 1, 5.999)

            mu = 2 * tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.tanh, trainable=trainable, scope="mu")
            mu = tf.clip_by_value(mu, 1, 5.999)

            sigma = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.softplus, trainable=trainable, scope="sigma")
            sigma = tf.clip_by_value(sigma, 1, 5.999)

            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma, allow_nan_stats=False)

        param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

        return (norm_dist, param)

    def build_actor_net_simple(self, scope, trainable):

        with tf.variable_scope(scope):

            reshaped_input = tf.reshape(self.s, [-1, 11, 11])
            flatten_input = tf.contrib.layers.flatten(reshaped_input)

            dl1 = tf.contrib.layers.fully_connected(inputs=flatten_input, num_outputs=200, activation_fn=tf.nn.relu, trainable=trainable, scope="simple_dl1")

            mu = 2 * tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.tanh, trainable=trainable, scope="simple_mu")
            mu = tf.clip_by_value(mu, 1, 5.999)

            sigma = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.softplus, trainable=trainable, scope="simple_sigma")
            sigma = tf.clip_by_value(sigma, 1, 5.999)

            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma, allow_nan_stats=False)

        param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

        return (norm_dist, param)

# class MlpPPO(BasePPO):
#
#     def __init__(self, action_space, observation_space, scope, args):
#
#         super().__init__(action_space, observation_space, scope, args)
#         self.build_net()
#
#     def _inception_block(self, scope, num_outputs=100, trainable=False):
#
#         with tf.variable_scope(scope):
#
#             board = tf.manip.reshape(self.s, [-1, 11, 11, 3], name="reshaped_input")
#
#             # initialize weights
#             filter_one = tf.get_variable("w01", [1, 1, 3, num_outputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)
#             filter_one_1 = tf.get_variable("w11", [1, 1, 3, num_outputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)
#             filter_one_2 = tf.get_variable("w12", [1, 1, 3, num_outputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)
#
#             filter_two_1 = tf.get_variable("w21", [1, 1, num_outputs, num_outputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)
#             filter_two_2 = tf.get_variable("w22", [1, 1, num_outputs, num_outputs], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)
#             # filter_two_3 = tf.get_variable("w23", [1, 1, 11, 11], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=trainable)
#
#
#             # description of inception block
#             final_one = tf.nn.conv2d(input=board, filter=filter_one, strides=[1, 1, 1, 1], padding="SAME", name="one_final")
#
#             # part one
#             stage_one_1 = tf.nn.conv2d(input=board, filter=filter_one_1, strides=[1, 1, 1, 1], padding="SAME", name="stage_one_1")
#             stage_one_2 = tf.nn.conv2d(input=board, filter=filter_one_2, strides=[1, 1, 1, 1], padding="SAME", name="stage_one_2")
#             # stage_one_max_pool = tf.layers.max_pooling2d(inputs=self.board, pool_size=[2, 2], strides=(1, 1), padding="SAME", name="stage_one_max_pool")
#
#             # part two
#             stage_two_1 = tf.nn.conv2d(input=stage_one_1, filter=filter_two_1, strides=[1, 1, 1, 1], padding="SAME", name="stage_two_1")
#             stage_two_2 = tf.nn.conv2d(input=stage_one_2, filter=filter_two_2, strides=[1, 1, 1, 1], padding="SAME", name="stage_two_2")
#             # stage_two_3 = tf.nn.conv2d(input=stage_one_max_pool, filter=filter_two_3, strides=[1, 1, 1, 1], padding="SAME", name="stage_two_3")
#
#             # merge
#             concat = tf.concat([final_one, stage_two_1, stage_two_2], axis=0, name="inception_block")
#             inception = tf.nn.relu(concat)
#             inception_flatten = tf.contrib.layers.flatten(inception)
#
#         return (inception_flatten)
#
#     def build_critic_net(self, scope):
#
#         with tf.variable_scope(scope):
#
#             board = tf.contrib.layers.flatten(self.s)
#
#             dl1 = tf.contrib.layers.fully_connected(inputs=board, num_outputs=100, activation_fn=tf.nn.relu, scope='dl1')
#
#             value = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=1, activation_fn=None, scope="value")
#
#             return (value)
#
#     def build_actor_net(self, scope, trainable):
#
#         inception_flatten = self._inception_block(scope, num_outputs=200, trainable=trainable)
#
#         with tf.variable_scope(scope):
#
#             dl1 = tf.contrib.layers.fully_connected(inputs=inception_flatten, num_outputs=200, activation_fn=tf.nn.relu, trainable=trainable, scope="dl1")
#
#             dl1 = tf.clip_by_value(dl1, 1, 5.999)
#
#             mu = 2 * tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.tanh, trainable=trainable, scope="mu")
#             mu = tf.clip_by_value(mu, 1, 5.999)
#
#             sigma = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.softplus, trainable=trainable, scope="sigma")
#             sigma = tf.clip_by_value(sigma, 1, 5.999)
#
#             norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma, allow_nan_stats=False)
#
#         param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
#
#         return (norm_dist, param)
