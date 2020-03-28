from utils import *
from model import A2C
from trainer import Trainer, Tester
import os
from easydict import EasyDict as edict
from pprint import pprint
import argparse, json


def parse_args():
    parser = argparse.ArgumentParser(description="A2C TensorFlow Implementation")
    parser.add_argument('--config_file', default="config/pong.json", type=str, help='Configuration file')
    parser.add_argument('--env_name', type=str, required=False,
                        default='PongNoFrameskip-v4', help="env")
    parser.add_argument('--is_training', type=str, required=False,
                        default=True, help="True=train, False=evaluation")
    parser.add_argument('--after_test', type=bool, required=False,
                        default=True, help="True=test after training")
    args = parser.parse_args()
    return args


def train_fn(args):
    train_summaries = args.summary_dir + '/train_summaries'
    os.makedirs(train_summaries, exist_ok=True)
    summary_writer = tf.summary.FileWriter(train_summaries)

    env_class = env_name_parser(args.env_class)
    env = make_all_environments(num_envs=args.num_envs, env_class=env_class, env_name=args.env_name,
                                seed=args.env_seed)
    model = A2C(config_args, env)
    trainer = Trainer(args, model, env, summary_writer)
    trainer.run()


def evaluate_fn(config_args):
    env_class = env_name_parser(config_args.env_class)
    env = make_all_environments(num_envs=1, env_class=env_class, env_name=config_args.env_name,
                                seed=config_args.env_seed)
    model = A2C(config_args, env)
    tester = Tester(config_args, model, env)
    tester.run()


if __name__ == '__main__':
    # Parse the JSON arguments
    args = parse_args()
    with open(args.config_file, 'r') as config_file:
        config_args_dict = json.load(config_file)
    config_args = edict(config_args_dict)
    pprint(config_args)
    print("\n")

    # Prepare Directories
    config_args.experiment_dir, config_args.summary_dir, config_args.checkpoint_dir, config_args.output_dir, config_args.test_dir = \
        create_experiment_dirs(config_args.experiment_dir)

    if args.is_training:
        train_fn(config_args)
    else:
        evaluate_fn(config_args)
