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

# saver = tf.train.import_meta_graph('/home/tools/Tools/raoqiang/facenet/models/facenet/20170807-231648/model-20170807-231648-0.meta')
#
# saver.restore(sess,tf.train.latest_checkpoint('/home/tools/Tools/raoqiang/facenet/models/facenet/20170807-231648/'))

def main(args):

    with tf.Session() as sess:
        tfa = TensorFlowAgent(name="TFA", args=args, sess=sess)
        tfa.restore_weigths(sess)


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
    parser.add_argument("--train", type=str, default="True")

    args = parser.parse_args()

    main(args)