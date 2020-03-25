import datetime
import numpy as np
import pickle
import tensorflow as tf
from common import logger


class Trainer():
    def __init__(self, args, model, env, summary_writer=None):
        self.args = args
        self.sess = model.sess
        self.cur_iteration = 0
        self.save_every = self.args.save_every
        self.summary_writer = summary_writer

        self.env = env
        self.model = model
        self.num_iterations = int(self.args.num_iterations)
        self.num_steps = self.model.num_steps

        self.model.init_model()
        self.model.load_model()

    def run(self):
        print('Training...')
        try:
            # Produce video only if monitor method is implemented.
            try:
                if self.args.record_video_every != -1:
                    self.env.monitor(is_monitor=True, is_train=True, experiment_dir=self.args.experiment_dir,
                                     record_video_every=self.args.record_video_every)
            except:
                pass

            self.global_time_step = self.model.global_time_step_tensor.eval(self.sess)

            # Calculate the batch_size
            nbatch = self.args.num_envs * self.num_steps

            for iteration in range(self.num_iterations // nbatch + 1):
                self.cur_iteration = iteration
                obs, states, rewards, masks, actions, values = self.model.forward()
                self.model.backward(obs, states, rewards, masks, actions,
                                    values, self.summary_writer, self.cur_iteration * nbatch)

                # Update the global step
                self.model.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.model.global_step_input: self.model.global_step_tensor.eval(self.sess) + 1})

                if not iteration % self.args.print_freq:
                    # mean_100ep_reward = round(np.mean(epoch_rewards[-99:-1]), 1)
                    # num_episodes = len(epoch_rewards)
                    logger.record_tabular("steps", iteration * nbatch)
                    # logger.record_tabular("episodes", num_episodes)
                    # logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("Current date and time: ", datetime.datetime.now())
                    logger.dump_tabular()

                if iteration % self.save_every == 0:
                    self.model.save()
            self.env.close()

        except KeyboardInterrupt:
            print('Error occured..\n')
            self.model.save()
            self.env.close()


class Tester():
    def __init__(self, args, model, env):
        self.args = args
        self.eval_num = self.args.eval_num
        self.env = env
        self.model = model

    def run(self):
        print('Training...')

        self.model.init_model()
        self.model.load_model()
        try:
            states = self.model.policy.actor_network.initial_state
            dones = [False for _ in range(self.env.num_envs)]

            observation_s = np.zeros(
                (self.env.num_envs, self.model.policy.img_height, self.model.policy.img_width,
                 self.model.policy.num_classes * self.model.policy.num_stack),
                dtype=np.uint8)
            observation_s = self.model.observation_update(self.env.reset(), observation_s)
            rewards = [0]
            for iteration in range(self.args.test_num):
                print("Test iteration = ", iteration)
                next = True
                while next:
                    actions, values, states = self.model.policy.actor_network.step(observation_s, states, dones)
                    observation, reward, dones, _ = self.env.step(actions)
                    rewards[-1] += reward
                    for n, done in enumerate(dones):
                        if done:
                            observation_s[n] *= 0
                            rewards.append(0)
                            next = False
                    observation_s = self.model.observation_update(observation, observation_s)
            print("Mean reward:", np.mean(rewards))
            self.env.close()
        except KeyboardInterrupt:
            print('Error occured..\n')
            self.env.close()
