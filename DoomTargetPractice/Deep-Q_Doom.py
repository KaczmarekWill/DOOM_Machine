import tensorflow as tf
import numpy as np
from vizdoom import *

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

"""
Here we create our environment
"""
def create_environment():
    game = DoomGame()

    # Load the correct configuration
    game.load_config("basic.cfg")

    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")

    game.init()

    # Here are our possible actions
    left  = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions

"""
Here we perform random actions to test the environment
"""
def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    left  = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions = [left, right, shoot]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()

game, possible_actions = create_environment()

def preprocess_frame(frame):
    # Greyscale frame already done in vizdoom config
    # x = np.mean(frame, -1)

    # Crop the top of the screen (remove the roof which contains no useful information
    cropped_frame = frame[30:-1,30:-30]

    # Normalize pixel values
    normalized_frame = cropped_frame/255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84,84])

    return preprocessed_frame

stack_size = 4

# Initialize deque with zero-valued images
stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear stacked frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4 times
        for i in range(4):
            stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append the frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

### MODEL HYPERPARAMETERS
state_size = [84,84,4] # Input is a stack of 4 84x84 frames
action_size = game.get_available_buttons_size() # 3 possible actions: left, right, shoot
learning_rate = 0.0002 # Alpha

### TRAINING HYPERPARAMETERS
total_episodes = 10000
max_steps = 100
batch_size = 64

# Exploration parameters for epsilon greedy strategy
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.00005

# Q learning hyperparameters
gamma = 0.95 # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size
memory_size = 50000

### MODIFY TO FALSE TO SEE TRAINED AGENT
training = True

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name='actions_')

            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(
                    inputs = self.inputs_,
                    filters = 32,
                    kernel_size = 8,
                    strides = 4,
                    padding = 'VALID',
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    name = 'conv1')

            self.conv1_batchnorm = tf.layers.batch_normalization(
                    self.conv1,
                    training = True,
                    epsilon = 1e-5,
                    name = 'batch_norm1')

            self.conv1_out = tf.nn.elu(
                    self.conv1_batchnorm,
                    name = 'conv1_out')
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 20x20x32
            self.conv2 = tf.layers.conv2d(
                    inputs = self.conv1_out,
                    filters = 64,
                    kernel_size = 4,
                    strides = 2,
                    padding = 'VALID',
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    name = 'conv2')

            self.conv2_batchnorm = tf.layers.batch_normalization(
                    self.conv2,
                    training = True,
                    epsilon = 1e-5,
                    name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(
                    self.conv2_batchnorm,
                    name = 'conv2_out')

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 9x9x64
            self.conv3 = tf.layers.conv2d(
                    inputs = self.conv2_out,
                    filters = 128,
                    kernel_size = 4,
                    strides = 2,
                    padding = 'VALID',
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    name = 'conv3')

            self.conv3_batchnorm = tf.layers.batch_normalization(
                    self.conv3,
                    training = True,
                    epsilon = 1e-5,
                    name = 'batch_norm3')

            self.conv3_out = tf.nn.elu(
                    self.conv3_batchnorm,
                    name = 'conv3_out')

            # Input is 3x3x128
            self.flatten = tf.layers.flatten(self.conv3_out)

            # Input is 1152x1
            self.fc = tf.layers.dense(
                    inputs = self.flatten,
                    units = 512,
                    activation = tf.nn.elu,
                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                    name = 'fc1')

            # Input is 512x1
            self.output = tf.layers.dense(
                    inputs = self.fc,
                    kernel_initializer = tf.contrib.layers.xavier_initializer(),
                    units = 3,
                    activation = None)

            # Q is our predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is the difference between our predicted Q and Q_target
            # Sum(Q_target - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)

        return [self.buffer[i] for i in index]

## Prepopulate the memory with random actions

# Instantiate memory
memory = Memory(max_size = memory_size)

# Render the environment
game.new_episode()

# Generate a state
state = game.get_state().screen_buffer
state, stacked_frames = stack_frames(stacked_frames, state, True)

for i in range(pretrain_length):
    # Random action
    action = random.choice(possible_actions)

    # Get rewards
    reward = game.make_action(action)

    done = game.is_episode_finished()

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))

        game.new_episode()

        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        memory.add((state, action, reward, next_state, done))
        state = next_state

# Set up TensorBoard Writer
writer = tf.summary.FileWriter('./tensorboard/1')

tf.summary.scalar('Loss', DQNetwork.loss)

write_op = tf.summary.merge_all()

### Train the agent
'''
Initialize the weights
Initialize the environment
Initialize the decay rate

For episode to max_episode do
    Make new episode
    Set step to 0
    Observe the first state

    While step < max_steps do
        Increase decay_rate
        Select random action or Q(s,a)
        Execute action and observe reward and new state
        Store transition
        Sample random mini-batch
        Set Qhat = r if episode ands at +1, otherwise Qhat = r+gamma*maxQ(s',a')
        Make a gradient descent step with loss(Qhat - Q(s,a))^2
'''
def predict_action(max_epsilon, min_epsilon, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * decay_step)
    
    if explore_probability > exp_exp_tradeoff:
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
        
    return action, explore_probability

saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate
        decay_step = 0

        game.init()

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1
                decay_step += 1

                action, explore_probability = predict_action(max_epsilon, min_epsilon, decay_rate, decay_step, state, possible_actions)
                reward = game.make_action(action)
                done = game.is_episode_finished()
                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step to max_steps to end the episode
                    step = max_steps

                    memory.add((state, action, reward, next_state, done))

                else:
                    next_state = game.get_state().screen_buffer
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    memory.add((state, action, reward, next_state, done))
                    state = next_state

                if done or step >= max_steps:
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))


                ### LEARNING PART
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []
                
                # Get Q values for next state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise r+gamma*maxQ(s',a')
                for i in range(len(batch)):
                    # If in terminal state, Q_target = reward
                    if dones_mb[i]:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.inputs_: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb})
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            #Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, './models/model.ckpt')
                print('Model Saved')

with tf.Session() as sess:
    game, possible_actions = create_environment()
    totalScore = 0

    saver.restore(sess, './models/model.ckpt')
    game.init()

    for i in range(100):
        done = False
        game.new_episode()

        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not done:
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()
            time.sleep(0.02)

            if done:
                break
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        totalScore += score
        print("Score: ", score)
    print("Average Score: ", totalScore / 100)
    game.close()
