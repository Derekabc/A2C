from agents.a2c_policy import ACPolicy
import tensorflow as tf
import numpy as np
from common.env_summary_logger import EnvSummaryLogger
from utils import create_list_dirs, LearningRateDecay
import pickle


class A2C:
    def __init__(self, args, env, lr_decay_method='linear'):
        self.args = args
        self.env = env
        self.gamma = self.args.gamma
        self.num_steps = self.args.unroll_time_steps

        self.global_time_step = 0
        self.num_iterations = int(self.args.num_iterations)

        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.policy = ACPolicy(self.sess,
                               optimizer_params={
                                   'learning_rate': args.learning_rate, 'alpha': 0.99, 'epsilon': 1e-5}, args=self.args)

        self.env_summary_logger = EnvSummaryLogger(self.sess,
                                                   create_list_dirs(self.args.summary_dir, 'env', self.args.num_envs))

        self.learning_rate_decayed = LearningRateDecay(v=self.args.learning_rate,
                                                       nvalues=self.num_iterations * self.args.unroll_time_steps * self.args.num_envs,
                                                       lr_decay_method=lr_decay_method)

        print("\n\nBuilding the model...")
        self.policy.init_policy(self.env.observation_space.shape, self.env.action_space.n)
        print("Model is built successfully\n\n")

        with open(self.args.experiment_dir + self.args.env_name + '.pkl', 'wb') as f:
            pickle.dump((self.env.observation_space.shape, self.env.action_space.n), f, pickle.HIGHEST_PROTOCOL)

        self.observation_s = np.zeros(
            (self.env.num_envs, self.policy.img_height, self.policy.img_width,
             self.policy.num_classes * self.policy.num_stack), dtype=np.uint8)
        self.observation_s = self.observation_update(self.env.reset(), self.observation_s)
        self.states = self.policy.actor_network.initial_state
        self.dones = [False for _ in range(self.env.num_envs)]

    def save(self):
        self.saver.save(self.sess, self.args.checkpoint_dir, self.global_step_tensor)

    def load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")
        else:
            print("No checkpoints available!\n\n")

    def init_global_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)

    def init_model(self):
        # init the global step, global time step, the current epoch and the summaries
        self.init_global_step()
        self.init_global_time_step()
        self.init_cur_epoch()
        self.init_global_saver()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def init_cur_epoch(self):
        """
        Create cur epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def init_global_time_step(self):
        """
        Create a global time step variable to be a reference to the number of time steps
        :return:
        """
        with tf.variable_scope('global_time_step'):
            self.global_time_step_tensor = tf.Variable(0, trainable=False, name='global_time_step')
            self.global_time_step_input = tf.placeholder('int32', None, name='global_time_step_input')
            self.global_time_step_assign_op = self.global_time_step_tensor.assign(self.global_time_step_input)

    def forward(self):
        """ run for n steps"""
        train_input_shape = (self.policy.train_batch_size, self.policy.img_height, self.policy.img_width,
                             self.policy.num_classes * self.policy.num_stack)

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states

        for n in range(self.num_steps):
            # Choose an action based on the current observation
            actions, values, states = self.policy.actor_network.step(self.observation_s, self.states, self.dones)

            # Actions, Values predicted across all parallel environments
            mb_obs.append(np.copy(self.observation_s))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take a step in the real environment
            observation, rewards, dones, info = self.env.step(actions)

            # Tensorboard dump, divided by 100 to rescale (to make the steps make sense)
            self.env_summary_logger.add_summary_all(int(self.global_time_step / (self.num_steps * self.args.num_envs)),
                                                    info)
            self.global_time_step += 1
            self.global_time_step_assign_op.eval(session=self.sess, feed_dict={
                self.global_time_step_input: self.global_time_step})

            # States and Masks are for LSTM Policy
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.observation_s[n] *= 0
            self.observation_s = self.observation_update(observation, self.observation_s)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(train_input_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.policy.actor_network.value(self.observation_s, self.states, self.dones).tolist()

        # Discount/bootstrap off value fn in all parallel environments
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = self.discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = self.discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # Instead of (num_envs, time_steps). Make them num_envs*time_steps.
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

    def backward(self, observations, states, rewards, masks, actions, values, summary_writer, global_step):
        # Updates the model per trajectory for using parallel environments. Uses the train_policy.
        advantages = rewards - values
        for step in range(len(observations)):
            current_learning_rate = self.learning_rate_decayed.value()

        # if states != []:
        # Leave it for now. It's for LSTM policy.
        # feed_dict[self.policy.S] = states
        # feed_dict[self.policy.M] = masks
        loss, policy_loss, value_loss, policy_entropy = self.policy.backward(self.sess, observations, actions, rewards,
                                                                             advantages, current_learning_rate,
                                                                             summary_writer, global_step)
        return loss, policy_loss, value_loss, policy_entropy

    def observation_update(self, new_observation, old_observation):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        updated_observation = np.roll(old_observation, shift=-1, axis=3)
        updated_observation[:, :, :, -1] = new_observation[:, :, :, 0]
        return updated_observation

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]
