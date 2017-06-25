# encoding: utf-8

import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


batch_size = 20
hidden_size = 64
learning_rate = 0.0001

memory_size = 10000
pretrain_length = batch_size

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

gamma = 0.99
max_step = 200
train_episodes = 1000

env = gym.make('CartPole-v0')

class Memory(object):
	def __init__(self,max_size=1000):
		self.buffer = deque(maxlen=max_size)
	def add(self,experience):
		self.buffer.append(experience)
	def sample(self,batch_size):
		idx = np.random.choice(np.arange(len(self.buffer)),size=batch_size,replace=False)
		return [self.buffer[i] for i in idx]
	def populate(self,env,pretrain_length):
		env.reset()
		state,_,_,_ = env.step(env.action_space.sample())
		for _ in range(pretrain_length):
			action = env.action_space.sample()
			next_state,reward,done,_ = env.step(action)
			if done:
				next_state = np.zeros(state.shape)
				self.add((state,action,reward,next_state))
				env.reset()
				state,_,_,_ = env.step(env.action_space.sample())
			else:
				self.add((state,action,reward,next_state))
				state = next_state

def sample_or_get_Qmax_action(env,step,Qmax_fn,state):
	explore_p = explore_stop + (explore_start-explore_stop)*np.exp(-decay_rate*step)
	if np.random.random() < explore_p:
		return env.action_space.sample(),explore_p
	else:
		return Qmax_fn(state),explore_p

def DBN(input_,out_size):
	fc1 = tf.contrib.layers.fully_connected(input_,hidden_size)
	fc2 = tf.contrib.layers.fully_connected(fc1,hidden_size)
	out = tf.contrib.layers.fully_connected(fc2,out_size,activation_fn=None)
	return out

def visualize(rewards_list):
	def running_mean(x, N):
	    cumsum = np.cumsum(np.insert(x, 0, 0)) 
	    return (cumsum[N:] - cumsum[:-N]) / N
	eps, rews = np.array(rewards_list).T
	smoothed_rews = running_mean(rews, 10)
	plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
	plt.plot(eps, rews, color='grey', alpha=0.3)
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')

class QNet(object):
	def __init__(self,env,memory,neural_net,max_step,learning_rate,state_size,action_size,name):
		self.sess = tf.Session()
		self.saver = None
		self.name = name
		self.env = env
		self.memory = memory
		self.neural_net = neural_net
		self.max_step = max_step
		self.global_step=0
		
		with tf.variable_scope(name):
			# batch_size x state_size
			self.inputs_ = tf.placeholder(shape=[None,state_size],dtype=tf.float32,name='inputs')
			# batch_size
			self.actions_ = tf.placeholder(shape=[None],dtype=tf.int32,name='actions')
			# batch_size x action_size
			self.one_hot_actions = tf.one_hot(self.actions_,action_size)
			# batch_size
			self.Q_hat = tf.placeholder(tf.float32,shape=[None],name='target')
			# hidden_size x action_size
			self.output = self.neural_net(self.inputs_,action_size)
			# batch_size
			self.Q = tf.reduce_sum(tf.multiply(self.output,self.one_hot_actions),axis=1)
			self.loss = tf.reduce_sum(tf.square(self.Q-self.Q_hat))
			self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	def q_look_up(self,state):
		'''
		being at a state s_i, what are the estimated q values for possible actions
		this intermediate result is usually skipped
		'''
		feed = {self.inputs_:state.reshape([1,*state.shape])}
		Qs = self.sess.run(self.output,feed_dict=feed)
		return Qs

	def get_maxQ_action(self,state):
		Qs = self.q_look_up(state)
		return np.argmax(Qs)

	def train_neural_net(self,states,actions,rewards,next_states):
		Q_t1= self.sess.run(self.output,feed_dict={self.inputs_:next_states})
		episode_ends = (next_states==0).all(axis=1)
		Q_t1[episode_ends] = (0,0)

		Q_t_hat = rewards + gamma * np.max(Q_t1,axis=1)
		loss,opt = self.sess.run([self.loss,self.opt],feed_dict={self.Q_hat:Q_t_hat,\
																self.inputs_:states,\
																self.actions_:actions})
		return loss

	def one_episode(self):
		self.env.reset()
		state,_,_,_ = self.env.step(self.env.action_space.sample())
		total_reward = 0
		t = 0
		while True:
			self.global_step += 1
			action,explore_p = sample_or_get_Qmax_action(self.env,self.global_step,self.get_maxQ_action,state)
			next_state,reward,done,_ = env.step(action)
			total_reward += reward
			
			# episode is done
			if done:
				next_state = np.zeros(state.shape)
				memory.add((state,action,reward,next_state))
				return total_reward,explore_p,loss
			# reach maximum allowed step, does not return reward
			elif t>= self.max_step:
				return None
			else:
				self.memory.add((state,action,reward,next_state))
				state = next_state
				t += 1
			batch = self.memory.sample(batch_size)
			states = np.array([each[0] for each in batch])
			actions = np.array([each[1] for each in batch])
			rewards = np.array([each[2] for each in batch])
			next_states = np.array([each[3] for each in batch])
			loss = self.train_neural_net(states,actions,rewards,next_states)

	def train_with_experience(self,n_episodes):
		rewards_list = []
		with self.sess as sess:
			sess.run(tf.global_variables_initializer())
			for ep_num in range(1,n_episodes+1):
				res = self.one_episode()
				if res is None:
					print('Episode: {} unfinished at step {}'.format(ep_num,self.max_step))
				else:
					reward,explore_p,loss = res
					rewards_list.append((ep_num,reward))
					print('Episode: {}; Total reward: {}; Exploration p {}; loss {} '.format(ep_num,reward,explore_p,loss))
			self.saver = tf.train.Saver()
			self.saver.save(sess,'checkpoints/cartpole.ckpt')
		return rewards_list

memory = Memory(max_size=memory_size)
memory.populate(env,pretrain_length)
state_size = env.reset().size
action_size = env.action_space.n
qnet = QNet(env,memory,DBN,max_step,learning_rate,state_size,action_size,'CartPole_DQN')
rewards_list = qnet.train_with_experience(train_episodes)
visualize(rewards_list)
