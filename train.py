import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from network.trainer import *
from argparse import ArgumentParser
from utils import read_config
import random

def train(settings):
    random.seed(42)
 
    trainer = Trainer(settings)
    trainer.train()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='', help='Which config to read')
    parser.add_argument('--check_point', type=str, default='', help='check point')
    args = parser.parse_args()

    settings = read_config(args.config)

    if args.check_point != '':
        settings['checkpoint'] = args.check_point

    try:
        train(settings)
    except Exception as e:
        print("Training task:%s crashed when error occur:\n%s"%(settings["job_name"], e), "Training task:%s crashed"%(settings["job_name"]))
        raise e
    print("Training task finish", "Training task:%s finish"%(settings["job_name"]))