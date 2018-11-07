# pommerman/cli/run_battle.py
# pommerman/agents/TensorFlowAgent/pit.py

import atexit
from datetime import datetime
import os
import random
import sys
import time

import argparse
import numpy as np

from pommerman import helpers, make
from TensorFlowAgent import TensorFlowAgent

from pommerman import utility

import tensorflow as tf

class Pit(object):

    def __init__(self, tfa, saver, game_nums=2):

        self.tfa = tfa
        self.saver = saver
        self.game_nums = game_nums

    def launch_games(self, sess, render=True):

        sess.run(tf.global_variables_initializer())
        self.tfa.restore_weigths(sess, self.saver)
        env = self.tfa.getEnv()

        reward_board = np.zeros((1, 4))

        for i in range(self.game_nums):

            curr_state = env.reset()

            while True:

                if render:
                    env.render()

                all_actions = env.act(curr_state)
                next_state, reward, terminal, _ = env.step(all_actions)

                if terminal:

                    reward_board += np.array(reward)

                    print("Game #{0}, rewards = {1}, reward agent = {2}".format(i, "".join(str(i) + " " for i in reward), reward[self.tfa.agent_id]))
                    break


def main(args):

    tf.reset_default_graph()

    with tf.Session() as sess:

        tfa = TensorFlowAgent(name="TFA", args=args, sess=sess)
        saver = tf.train.Saver(allow_empty=True)
        pit = Pit(tfa, saver, game_nums=2)

        pit.launch_games(sess)



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
    parser.add_argument("--train", type=str, default="False", choices=["False"])
    parser.add_argument("--type", type=str, default="Simple", choices=["Simple, CNN"])

    args = parser.parse_args()

    main(args)