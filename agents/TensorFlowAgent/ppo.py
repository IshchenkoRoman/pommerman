import tensorflow as tf
import numpy as np
import os

class BasePPO(object):

    def __init__(self, action_space, observation_space, scope, args):
        self.action_space = action_space
        self.observation_space = observation_space
        self.scope = scope
        self.action_bound = [0.001, 4.999]
        self.num_state = self.observation_space
        self.num_action = 1
        self.cliprange = args.cliprange
        self.checkpoint_path = "".join([args.checkpoint_dir, "/", args.environment, "/", args.policy])
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        with tf.variable_scope("input"):
            self.s = tf.placeholder("float", [None, self.num_state])
        with tf.variable_scope("action"):
            self.a = tf.placeholder(shape=[None, self.num_action], dtype=tf.float32)
        with tf.variable_scope("target_value"):
            self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope("advantages"):
            self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    def build_critic_net(self, net):

        raise NotImplementedError("You cann't instantiate this class")

    def build_actor_net(self, scope, trainable):

        raise NotImplementedError("You cann't instantiate this class")

    def build_net(self):

        self.value = self.build_critic_net("value_net")
        pi, pi_param = self.build_actor_net("actor_net", trainable=True)
        old_pi, old_pi_param = self.build_actor_net("old_actor_net", trainable=False)
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

        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("...................Model restored to global..............")
        else:
            print("..................No model is found....................")

    def save_model(self, sess, saver, time_step):
        print("....................save model...................")
        saver.save(sess, "".join([self.checkpoint_path, "/", "Pommerman", "-", str(time_step), ".ckpt"]))

    def choose_action(self, s, sess):

        s = s[np.newaxis, :]
        a = sess.run(self.sample_op, {self.s: s})

        return (a)

    def get_v(self, s, sess):
        if s.ndim < 2 : s = s[np.newaxis, :]
        return (sess.run(self.value, {self.s: s})[0,0])

class MlpPPO(BasePPO):

    def __init__(self, action_space, observation_space, scope, args):

        super().__init__(action_space, observation_space, scope, args)
        self.build_net()

    def build_critic_net(self, scope):

        with tf.variable_scope(scope):

            dl1 = tf.contrib.layers.fully_connected(inputs=self.s, num_outputs=100, activation_fn=tf.nn.relu, scope='dl1')

            value = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=1, activation_fn=None, scope="value")

            return (value)

    def build_actor_net(self, scope, trainable):

        with tf.variable_scope(scope):
            dl1 = tf.contrib.layers.fully_connected(inputs=self.s, num_outputs=200, activation_fn=tf.nn.relu, trainable=trainable, scope="dl1")

            dl1 = tf.clip_by_value(dl1, 1, 4.999)

            mu = 2 * tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.tanh, trainable=trainable, scope="mu")
            mu = tf.clip_by_value(mu, 1, 4.999)

            sigma = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=6, activation_fn=tf.nn.softplus, trainable=trainable, scope="sigma")
            sigma = tf.clip_by_value(sigma, 1, 4.999)

            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma, allow_nan_stats=False)

        param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

        return (norm_dist, param)
