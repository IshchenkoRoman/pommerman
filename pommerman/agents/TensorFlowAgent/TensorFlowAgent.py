import os
import glob
import argparse

import tensorflow as tf
import numpy as np

import pommerman

from pommerman.agents import BaseAgent
from pommerman.agents import SimpleAgent

from pommerman import characters
from pommerman import constants

from ppo import MlpPPO

NUM_AGENTS = 4

BOARD_SIZE = constants.BOARD_SIZE
BOARD_SIZE_SQ = BOARD_SIZE  * BOARD_SIZE

class TensorFlowAgent(BaseAgent):
    """TensorFlowAgent for game pommerman.

        This agent used PPO algorithm with some modifications: on input of actor can used as
        flatten board or as matrix 11x11. In second case before counting normal distribution
        used CNN for extracting features.

        Parameters:
            name: str
                A string, name of agent.
            sess: tensorflow.python.client.session.Session
                Current tf.Session, used for restoring and train weights.
            character: Bomber
                Used for parent constructor.
            aget_id: int
                agent position in initialised list of gym.environment. Also used for get
                data form environment observation, while training.
    """

    def __init__(self, name, args, sess, character=characters.Bomber, agent_id=0):
        super(TensorFlowAgent, self).__init__(character)

        self.args = args

        # Pathes where weights saved
        self.curr_path = os.path.dirname(os.path.realpath(__file__))
        self.path_models = "".join([self.curr_path, "/", self.args.checkpoint_dir, "/", self.args.environment, "/", self.args.policy])

        # Work data agent id, agent_id representation in matrix observation
        self.agent_id = agent_id
        self.agent_id_dec = agent_id + 10

        self.scope = name #depricated
        self.trainable = args.train # Train or play mode
        self.type = args.type # Type of NN for PPO(Simple- only flatten, CNN- convolution)

        self.env = self._make_env() # Make gym environment, add agents to game
        self.action_space = 6 # Count of all actions, that agent can make
        self.observation_space = 121 # flatten size of observation

        # Init PPO algorithm
        self.ppo = MlpPPO(self.action_space, self.observation_space, self.scope, args, self.path_models, type=self.type)

        # Some logs
        self.summary_writer = tf.summary.FileWriter(args.summary_dir + '/' + args.environment + '/' + args.policy)
        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increase_global_episodes = self.global_episodes.assign_add(1)

        # Params of clipping
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.training_step = args.training_step
        self.train_op_actor = tf.train.AdamOptimizer(args.a_learning_rate).minimize(self.ppo.actor_loss)
        self.train_op_critic = tf.train.AdamOptimizer(args.c_learning_rate).minimize(self.ppo.critic_loss)

        self.imported_graph = None
        self.sess = sess

    def _init_input_for_game(self):

        """ Init output and input nodes for NN prediction."""
        if self.trainable == "False":

            self.prediction_actions = tf.get_default_graph().get_tensor_by_name("prediction:0")

            self.input_observation = tf.get_default_graph().get_tensor_by_name(self.type + "_input_observation/state:0")
            self.input_ammos = tf.get_default_graph().get_tensor_by_name(self.type + "_input_ammos/input_ammos:0")
            self.input_alive_agents = tf.get_default_graph().get_tensor_by_name(self.type + "_alive_agents/alive_agents:0")

            return (self.prediction_actions, self.input_observation, self.input_ammos, self.input_alive_agents)

    def getEnv(self):

        return (self.env)

    def _make_env(self):
        """Initialise gym environment, adding agents to them."""
        agents = [self if agent_id == self.agent_id else SimpleAgent() for agent_id in range(NUM_AGENTS)]

        return (pommerman.make("PommeFFACompetition-v0", agents))

    def _form_observation_agent(self, curr_state):

        """Get from observation for all agent, data for desirable/training agent.
        Parameters:
            curr_state: 2d list of dictionaries [{}]
                Observation for all agents in game.
        """
        curr_states = curr_state[self.agent_id]
        board = np.array(curr_states["board"])
        ammo = curr_states["ammo"]

        return (board, ammo)

    def restore_weigths(self, sess, saver):

        """Restore our model with wegths from .meta file."""
        list_of_weigths = [f for f in glob.glob(self.path_models + "/" + "*.meta")]
        if not list_of_weigths:
            raise (Exception("No '.meta' files!"))
        list_of_weigths.sort()

        self.imported_graph = tf.train.import_meta_graph(list_of_weigths[-1])
        saver.restore(sess, tf.train.latest_checkpoint(self.path_models))

    def _process_terminal(self, global_episodes, episode_length, total_rewards, reward):

        """Logging if environment get terminal state"""
        print("AR = ", self.agent_reward, reward)
        print('ID :' + self.scope + ', global episode :' + str(
            global_episodes) + ', episode length :' + str(episode_length) + ', total reward :' + str(total_rewards))
        summary = tf.Summary()
        summary.value.add(tag='Rewards/Total_Rewards', simple_value=float(total_rewards))
        summary.value.add(tag='Rewards/Episode_Length', simple_value=float(episode_length))
        self.summary_writer.add_summary(summary, global_episodes)
        self.summary_writer.flush()

    def process(self, sess, saver, render=False):

        """Main magic VooDoo. Partly of PPO paper present here: clipping values, form batches for training.
        Parameters:
            sess: tensorflow.python.client.session.Session
                current session.
            saver:tensorflow.python.client.train.Saver
                Used for saving weights if agent win or reached condition.
            render: bool
                If True- show game board.
        """
        curr_state = self.env.reset()
        total_rewards = 0
        episode_length = 0
        self.num_training = 0

        while True:
            states_buf = []
            actions_buf = []
            rewards_buf = []

            for i in range(0, self.batch_size):

                global_episodes = sess.run(self.increase_global_episodes)

                if render:
                    self.env.render()

                all_actions = self.env.act(curr_state)

                updated_input, ammo = self._form_observation_agent(curr_state)

                action = self.ppo.choose_action(updated_input, ammo, sess)
                action = np.argmax(np.int32(action))
                all_actions[self.agent_id] = action
                next_state, reward, self.terminal, _ = self.env.step(all_actions)

                self.agent_reward = reward[self.agent_id]

                total_rewards += self.agent_reward
                episode_length += 1
                states_buf.append(updated_input)
                actions_buf.append(action)
                rewards_buf.append(self.agent_reward)

                curr_state = next_state

                if self.terminal or self.agent_id_dec not in curr_state[self.agent_id]["alive"]:

                    self._process_terminal(global_episodes, episode_length, total_rewards, reward)
                    curr_state = self.env.reset()
                    total_rewards = 0
                    episode_length = 0
                    self.agent_reward = 0
                    break

                bootstrap_value = self.ppo.get_v(updated_input, sess)

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

        """Train module for PPO.
        Params:
            sess: tensorflow.python.client.session.Session
                current session
            saver:tensorflow.python.client.train.Saver
                Used for saving weights if agent win or reached condition.
            s: 2d list of strings
                batch with size 'self.batch_size' of states of previous states.
            a: list of ints
                batch with size 'self.batch_size' of states of previous actions.
            r: list of int
                batch with size 'self.batch_size' of states of previous rewards.
        """
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

        if self.agent_reward == 1:
            self.ppo.save_model(sess, saver, global_episodes)

    def choose_action(self, s):
        """Deprecated. See PPO.choose_action"""
        prob_weights = self.sess.run(self.pi, feed_dict={self.input_layer: s})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())

        return (action)

    def get_v(self, s):
        """Deprecated. See PPO.get_v"""
        if len(s.shape) < 2: s = s[np.newaxis, :]
        return (self.sess.run(self.v, {self.input_layer: s})[0, 0])

    def act(self, obs, action_space):

        """Function for make prediction of moves.
        This function works only with [False] '--train' flag.
        Parameters:
            obs: 2d list of dictionaries
                data for agent. Example of full description:
                    'teammate' = {Item} Item.AgentDummy
                    'alive' = {list} <class 'list'>: [10, 11, 12, 13]
                    'ammo' = {int} 1
                    'blast_strength' = {int} 2
                    'board' = {ndarray}
                    'bomb_blast_strength' = {ndarray}
                    'bomb_life' = {ndarray}
                    'can_kick' = {bool} False
                    'position' = {tuple} <class 'tuple'>: (1, 1)
                    'enemies' = {list} <class 'list'> [<Item.Agent1: 11>, <Item.Agent2: 12>, <Item.Agent3: 13>]
                    'game_env' = {str} 'pommerman.envs.v0:Pomme'
                    'game_type' = {int} 1
        """
        if self.trainable == "False":
            current_obs = obs["board"]
            ammo = obs["ammo"]

            prdeicted_actions, input_board, input_ammo, input_alive_agents = self._init_input_for_game()

            predict = np.array(self.sess.run(prdeicted_actions, feed_dict={input_board: current_obs, input_ammo: ammo}))
            return (np.argmax(np.trunc(predict)))

        pass

def main(args):

    tf.reset_default_graph()

    with tf.Session() as sess:

        tfa = TensorFlowAgent(name="TFA", args=args, sess=sess)
        saver = tf.train.Saver(allow_empty=True)
        sess.run(tf.global_variables_initializer())
        tfa.ppo.load_model(sess, saver)
        tfa.process(sess, saver, render=True)

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
    parser.add_argument("--train", type=str, default="True", choices=["True, False"])
    parser.add_argument("--type", type=str, default="Simple", choices=["Simple, CNN"])



    args = parser.parse_args()

    main(args)