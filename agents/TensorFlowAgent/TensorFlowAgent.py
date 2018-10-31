import pommerman

from pommerman.agents import BaseAgent
from pommerman.agents import SimpleAgent

from pommerman import characters
from pommerman import constants

import tensorflow as tf
from tensorflow.python.client import device_lib

import numpy as np
# import matplotlib.pyplot as plt
import gym

from ppo import MlpPPO

import argparse

NUM_AGENTS = 4

BOARD_SIZE = constants.BOARD_SIZE
BOARD_SIZE_SQ = BOARD_SIZE  * BOARD_SIZE

class TensorFlowAgent(BaseAgent):
    """TensorFlowAgent."""

    def __init__(self, name, args, character=characters.Bomber, agent_id=0):
        super(TensorFlowAgent, self).__init__(character)

        self.agent_id = agent_id
        self.scope = name

        self.env = self.make_env()
        self.action_space = 6
        # self.observation_space = self.env.observation_space.shape[0] // 4
        self.observation_space = 121

        self.ppo = MlpPPO(self.action_space, self.observation_space, self.scope, args)

        self.summary_writer = tf.summary.FileWriter(args.summary_dir + '/' + args.environment + '/' + args.policy)
        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increse_global_episodes = self.global_episodes.assign_add(1)

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.training_step = args.training_step
        self.train_op_actor = tf.train.AdamOptimizer(args.a_learning_rate).minimize(self.ppo.actor_loss)
        self.train_op_critic = tf.train.AdamOptimizer(args.c_learning_rate).minimize(self.ppo.critic_loss)

    def make_env(self):
        agents = [self if agent_id == self.agent_id else SimpleAgent() for agent_id in range(NUM_AGENTS)]

        return (pommerman.make("PommeFFACompetition-v0", agents))

    def process(self, sess, saver, render=False):

        curr_state = self.env.reset()
        # curr_state = self.env.getobservation()
        next_state = 0
        total_rewards = 0
        episode_length = 0
        global_episodes = 0
        self.num_training = 0

        while True:
            states_buf = []
            actions_buf = []
            rewards_buf = []

            for i in range(0, self.batch_size):

                global_episodes = sess.run(self.increse_global_episodes)

                if render:
                    self.env.render()

                all_actions = self.env.act(curr_state)
                curr_state_a = curr_state[self.agent_id]["board"].ravel()
                action = self.ppo.choose_action(curr_state_a, sess)
                action = np.max(np.int32(action))
                all_actions[self.agent_id] = action
                next_state, reward, self.terminal, _ = self.env.step(all_actions)

                # agent_state = self.env.featurize(next_state[self.agent_id])
                agent_state = curr_state[self.agent_id]["board"].ravel()
                agent_reward = reward[self.agent_id]

                total_rewards += agent_reward
                episode_length += 1
                states_buf.append(agent_state)
                actions_buf.append(action)
                rewards_buf.append(agent_reward)

                curr_state = next_state
                if self.terminal:
                    print('ID :' + self.scope + ', global episode :' + str(
                    global_episodes)+ ', episode length :' + str(episode_length)+ ', total reward :' + str(total_rewards))
                    curr_state = self.env.reset()
                    summary = tf.Summary()
                    summary.value.add(tag='Rewards/Total_Rewards', simple_value=float(total_rewards))
                    summary.value.add(tag='Rewards/Episode_Length', simple_value=float(episode_length))
                    self.summary_writer.add_summary(summary, global_episodes)
                    self.summary_writer.flush()
                    total_rewards = 0
                    episode_length = 0
                    break

                bootstrap_value = self.ppo.get_v(np.array(next_state[self.agent_id]["board"].ravel()), sess)

                if states_buf:
                    discounted_r = []
                    v_s_ = bootstrap_value
                    for r in rewards_buf[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)

                    discounted_r.reverse()
                    bs, ba, br = np.vstack(states_buf), np.vstack(actions_buf), np.array(discounted_r)[:, np.newaxis]
                    self.train(sess, saver, bs, ba, br)

    def train(self, sess, saver, s, a, r):

        sess.run(self.ppo.syn_old_pi)
        global_episodes = sess.run(self.global_episodes)
        self.num_training += 1

        adv = sess.run(self.ppo.adv, {self.ppo.s: s, self.ppo.y: r})
        feed_dict_actor = {}
        feed_dict_actor[self.ppo.s] = s
        feed_dict_actor[self.ppo.a] = a
        feed_dict_actor[self.ppo.advantage] = adv

        feed_dict_critic = {}
        feed_dict_critic[self.ppo.s] = s
        feed_dict_critic[self.ppo.y] = r

        [sess.run(self.train_op_actor, feed_dict=feed_dict_actor) for _ in range(self.training_step)]
        [sess.run(self.train_op_critic, feed_dict=feed_dict_critic) for _ in range(self.training_step)]

        if self.num_training % 500 == 0:
            self.ppo.save_model(sess, saver, global_episodes)

    def choose_action(self, s):
        prob_weights = self.sess.run(self.pi, feed_dict={self.input_layer: s})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

        return (action)

    def get_v(self, s):
        if len(s.shape) < 2: s = s[np.newaxis, :]
        return (self.sess.run(self.v, {self.input_layer: s})[0, 0])

    def act(self, obs, action_space):
        pass


def main(args):

    tf.reset_default_graph()

    tfa = TensorFlowAgent("TFA", args)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())
        tfa.ppo.load_model(sess, saver)
        tfa.process(sess, saver)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", type=str, default="pommerman")
    parser.add_argument("--policy", type=str, default="MlpPolicy")
    parser.add_argument("--checkpoint_dir", type=str, default="./save_model")
    parser.add_argument("--a_learning_rate", type=float, default=0.0001)
    parser.add_argument("--c_learning_rate", type=float, default=0.0002)
    parser.add_argument('--summary_dir', type=str, default='./summary_log')
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--training_step", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.9)

    args = parser.parse_args()

    main(args)