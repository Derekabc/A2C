import os, random
import tensorflow as tf
from envs.subproc_vec_env import *


def __env_maker(env_class, env_name, i, seed):
    def __make_env():
        return env_class(env_name, i, seed)

    return __make_env


def make_all_environments(num_envs=4, env_class=None, env_name="SpaceInvaders", seed=42):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return SubprocVecEnv(
        [__env_maker(env_class, env_name, i, seed) for i in range(num_envs)])


def env_name_parser(env_name):
    from envs.gym_env import GymEnv
    envs_to_class = {'GymEnv': GymEnv}

    if env_name in envs_to_class:
        return envs_to_class[env_name]
    raise ValueError("There is no environment with this name. Make sure that the environment exists.")


# def parse_args():
#     """
#     Parse the arguments of the program
#     :return: (config_args)
#     :rtype: tuple
#     """
#     # Create a parser
#     parser = argparse.ArgumentParser(description="A2C TensorFlow Implementation")
#     parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
#     parser.add_argument('--config', default="config/breakout.json", type=str, help='Configuration file')
#
#     # Parse the arguments
#     args = parser.parse_args()
#
#     # Parse the configurations from the config json file provided
#     try:
#         if args.config is not None:
#             with open(args.config, 'r') as config_file:
#                 config_args_dict = json.load(config_file)
#         else:
#             print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
#             exit(1)
#
#     except FileNotFoundError:
#         print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
#         exit(1)
#     except json.decoder.JSONDecodeError:
#         print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
#         exit(1)
#
#     config_args = edict(config_args_dict)
#
#     pprint(config_args)
#     print("\n")
#
#     return config_args


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = "experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    output_dir = experiment_dir + 'output/'
    test_dir = experiment_dir + 'test/'
    dirs = [summary_dir, checkpoint_dir, output_dir, test_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created")
        return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def create_list_dirs(input_dir, prefix_name, count):
    dirs_path = []
    for i in range(count):
        dirs_path.append(input_dir + prefix_name + '-' + str(i))
        create_dirs([input_dir + prefix_name + '-' + str(i)])
    return dirs_path


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


class LearningRateDecay(object):
    def __init__(self, v, nvalues, lr_decay_method):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues

        def constant(p):
            return 1

        def linear(p):
            return 1 - p

        lr_decay_methods = {
            'linear': linear,
            'constant': constant
        }

        self.decay = lr_decay_methods[lr_decay_method]

    def value(self):
        current_value = self.v * self.decay(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def get_value_for_steps(self, steps):
        return self.v * self.decay(steps / self.nvalues)
