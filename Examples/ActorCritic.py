'''
# Define the state of the Bipedal Walker. This is the information we will use to train the AI

state = [
self.hull.angle, # Varies from -0.5 to 0.5, where 0 is level
2.0 * self.hull.angularVelocity/FPS, # Normalized angular vel of hull
0.3 * vel.x * (VIEWPORT_W/SCALE)/FPS, # Normalized x velocity
0.3 * vel.y * (VIEWPORT_H/SCALE)/FPS, # Normalized y velocity
self.joints[0].angle,
self.joints[0].speed / SPEED_HIP,
self.joints[1].angle,
self.joints[1].speed / SPEED_KNEE,
1.0 if self.legs[1].ground_contact else 0.0,
self.joints[2].angle,
self.joints[2].speed / SPEED_HIP,
self.joints[3].angle,
self.joints[3].speed / SPEED_KNEE,
1.0 if self.legs[3].ground_contact else 0.0
]

# Translate the action inputs into rotation of the joints on the bot

self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))

'''

###### Imports ######

import gym
import numpy as np
import tensorflow as tf
import time
import os
import random
from collections import deque

from tensorflow import ConfigProto

###### Define model save/load files ######

# Update this to create model/log files for a new bot instance
suffix = '03'

# Update this to load a model from the specified iteration
sample_num = '6200'

score_save_name = './Bipedal_Logs/Score-{}.npy'.format(suffix)
model_save_name = './Bipedal_Models/Bipedal-{}'.format(suffix)
model_load_name = './Bipedal_Models/6200/Bipedal-{}-{}'.format(suffix, sample_num)

###### Parameters ######

save_model = True       # Save models during training
load_model = False

replay_count = 1000
render = True          # Render training iterations
max_game_step = 650     # Max steps to achieve target. This can speed up training if bot is getting stuck
use_gpu = 1             # Use gpu for training/testing
config = ConfigProto(device_count={'GPU': use_gpu})

tau = 0                 
tau_max = 5000          # Update target model after tau_max steps
lr_actor = 1e-4
lr_critic = 1e-3
# The actor-critic model has two components: an actor and a critic. The actor takes in the current environment state and determines the best action to take. The critic evaluates the environment state and action and returns a score that represents how apt the action is for the state.

N = 10000               # Number of training samples
test_num = 10           # Frequency to test/save model

gamma = 0.99            # Discount factor that is multiplied by the long-term reward. Counteracts delayed gratification

minibatch_size = 64     # Number of steps to feed the nn at a time
memory = deque(maxlen=500000)   # Store steps
pre_train_steps = 5000  # Number of steps to have in memory before sampling minibatch

# Generate the Walker environment
env = gym.make('BipedalWalker-v2')
input_size = env.observation_space.shape[0] # 24 observation values
output_size = env.action_space.shape[0] # 4 action values
t0 = time.time()
noise = np.zeros(output_size)

epsilon = 1
###### Functions ######
iteration = 0
def play_one(env, model, gamma):
    global iteration
    global epsilon
    max_epsilon = 1
    min_epsilon = .01
    decay_rate = .005
    
    state = env.reset()
    done = False
    totalreward = 0
    global tau
    # while not done
    for i in range(max_game_step):
        state = np.array(state).reshape(-1, input_size)
        exp_exp_tradeoff = random.uniform(0,1)
        if exp_exp_tradeoff > epsilon:
            action = sess.run(nn_actor.outputs, feed_dict={nn_actor.state_inputs: state}) # generate next action from actor network
            action = np.clip(action, -1, 1)
            action = action[0]
        else:
            action = env.action_space.sample()
        #noise = generateNoise()
        #action += noise
        #action = np.clip(action, -1, 1)
        #action = action[0]
        next_state, reward, done, info = env.step(action)

        totalreward += reward
        last_sequence = (state, action, reward, next_state, done)
        memory.append(last_sequence)

        state = next_state
        tau += 1

        if len(memory) > pre_train_steps:
            minibatch = random.sample(memory, minibatch_size)
            train(minibatch)

        if tau > tau_max:
            update_target_actor, update_target_critic = update_targets()
            sess.run([update_target_actor, update_target_critic])
            tau = 0
        
        if render == True:
            if iteration % 20 == 0:
                env.render()

        if done:
            noise = np.zeros(output_size)
            break
    
    print(totalreward)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*iteration)
    iteration += 1
    return totalreward

def replay(model, num, test=False):
    totalrewards = np.empty(replay_count)

    for game in range(num):
        state = env.reset()
        game_score = 0
        done = False
        while not done:
            state = np.array(state).reshape(-1, input_size)
            action = sess.run(nn_actor.outputs, feed_dict={nn_actor.state_inputs: state})[0]

            state, reward, done, info = env.step(action)
            game_score += reward

            if not test:
                env.render()
        
        if not test:
            print('Game {} score: {}'.format(game, game_score))
        totalrewards[game] = game_score
        average_10 = totalrewards[max(0, game-10):(game+1)].mean()
        output = 'Average score last 10: {}\n'.format(average_10)
        if game % 10 == 0 and game > 0:
            print(output)
    if test:
        return average_10

def update_targets():
    # get network parameters
    from_vars_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nn_actor')
    from_vars_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nn_critic')

    # get target parameters
    to_vars_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nn_actor_target')
    to_vars_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'nn_critic_target')

    holder_actor = []
    holder_critic = []

    # update target network parameters with network parameters
    for from_vars_actor, to_vars_actor in zip(from_vars_actor, to_vars_actor):
        holder_actor.append(to_vars_actor.assign(from_vars_actor))
    for from_vars_critic, to_vars_critic in zip(from_vars_critic, to_vars_critic):
        holder_critic.append(to_vars_critic.assign(from_vars_critic))

    return holder_actor, holder_critic

def generateNoise():
    return 0.2 * np.random.randn(output_size)

def train(minibatch):
    state_batch = np.asarray([data[0] for data in minibatch])
    action_batch = np.asarray([data[1] for data in minibatch])
    reward_batch = np.asarray([data[2] for data in minibatch])
    next_state_batch = np.asarray([data[3] for data in minibatch])
    done_batch = np.asarray([data[4] for data in minibatch])

    next_action_batch = sess.run(nn_actor_target.outputs, feed_dict={nn_actor_target.state_inputs: next_state_batch})

    q_value_batch = sess.run(nn_critic_target.output, feed_dict={nn_critic_target.state_inputs: next_state_batch, nn_critic_target.action_inputs: next_action_batch})
    y_batch = []

    # If done append reward only else append Discounted Qvalue
    for i in range(len(minibatch)):
        if done_batch[i]:
            y_batch.append(reward_batch[i])
        else:
            y_batch.append(reward_batch[i] + gamma * q_value_batch[i])

    state_batch = state_batch.reshape(-1, input_size)
    action_batch = action_batch.reshape(-1, output_size)

    y_batch = np.asarray(y_batch)
    y_batch = y_batch.reshape(-1, 1)

    # Train op
    _, loss, outputs = sess.run([nn_critic.train_op, nn_critic.loss, nn_critic.output], feed_dict={nn_critic.state_inputs: state_batch, nn_critic.action_inputs: action_batch, nn_critic.target_Q: y_batch})

    action_batch_for_grads = sess.run(nn_actor.outputs, feed_dict={nn_actor.state_inputs: state_batch})
    q_gradient_batch = sess.run(nn_critic.action_gradients, feed_dict={
            nn_critic.state_inputs: state_batch,
            nn_critic.action_inputs: action_batch_for_grads
        })[0]

    _, par_grads, train_vars = sess.run([nn_actor.optimizer, nn_actor.parameters_gradients, nn_actor.train_vars], feed_dict={nn_actor.state_inputs: state_batch, nn_actor.q_gradient_inputs: q_gradient_batch})

class ActorNet:
    def __init__(self, name, env=None, target=False):
        self.name = name
        self.env = env
        self.target = target

        # Actor build
        with tf.variable_scope(self.name):
            self.state_inputs = tf.placeholder(tf.float32, [None, 24], name="state_inputs")
            self.actor_l1 = tf.layers.dense(self.state_inputs, 256, activation=tf.nn.relu)
            self.actor_l2 = tf.layers.dense(self.actor_l1, 512, activation=tf.nn.relu)
            self.outputs = tf.layers.dense(self.actor_l2, output_size, activation=tf.nn.tanh)

            # Training stage
            if not target:
                self.q_gradient_inputs = tf.placeholder(tf.float32, [None, output_size])
                self.train_vars = tf.trainable_variables()

                self.parameters_gradients = tf.gradients(self.outputs, self.train_vars,
                                                         -self.q_gradient_inputs)

                self.optimizer = tf.train.AdamOptimizer(lr_actor).apply_gradients(
                     zip(self.parameters_gradients, self.train_vars))


class CriticNet:
    def __init__(self, name, env=None, target=False):
        self.name = name
        self.env = env

        # Critic build
        with tf.variable_scope(self.name):
            self.state_inputs = tf.placeholder(tf.float32, [None, 24], name="critic_state_inputs")

            self.action_inputs = tf.placeholder(tf.float32, [None, 4], name="critic_action_inputs")

            self.state_l1 = tf.layers.dense(self.state_inputs, 256, activation=tf.nn.relu)

            self.action_l1 = tf.layers.dense(self.action_inputs, 256, activation=tf.nn.relu)

            self.mergedlayer = tf.concat([self.state_l1, self.action_l1], 1)

            self.mergedlayer_l1 =  tf.layers.dense(self.mergedlayer, 512, activation=tf.nn.relu)

            self.output = tf.layers.dense(self.mergedlayer_l1, 1)

            if not target:
                # Training stage
                self.target_Q = tf.placeholder(tf.float32, [None, 1], name="target_Q")

                #self.loss = tf.losses.mean_squared_error(self.target_Q, self.outputs)
                self.loss = tf.losses.huber_loss(self.target_Q, self.output)

                self.train_op = tf.train.AdamOptimizer(learning_rate=lr_critic).minimize(self.loss)

                self.action_gradients = tf.gradients(self.output, self.action_inputs)
                # self.action_gradients = tf.gradients(self.output, self.state_inputs)

tf.reset_default_graph()

# Instantiate Actor Network
nn_actor = ActorNet(name='nn_actor', env=env)

# Instantiate Actor's Target Network
nn_actor_target = ActorNet(name='nn_actor_target', env=env, target=True)

# Instantiate Critic Network
nn_critic = CriticNet(name='nn_critic', env=env)

# Instantiate Critic's Target Network
nn_critic_target = CriticNet(name='nn_critic_target', env=env, target=True)

saver = tf.train.Saver()

if load_model:
    # Not worth using GPU for replaying model
    config_replay = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=config_replay) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, model_load_name)

        replay(nn_actor, replay_count)

        exit()

totalrewards = np.empty(N)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    scores_for_graph = []

    for n in range(N):
        # Play one game
        play_one(env, nn_actor, gamma)

        if n > 1 and n % 10 == 0:
            # Testing model
            avg_score = replay(nn_actor, test_num, test=True)

            if save_model:
                saver.save(sess, model_save_name, global_step=n)

            if avg_score >= 300:
                saver.save(sess, model_save_name, global_step=n)

env.close()
sess.close()
