import json
import os
from pickletools import uint8

class Settings:
    def __init__(self):
        self.job_name = ''

        self.jitter_samples = 8

        self.gpu = [0]

        self.upsample_factor = 4

        self.log_folder = "log/"

        self.check_data = True

        self.num_workers = 16

        self.model_name = ""
        self.model_file = ""
        self.model_input = [

        ]

        self.optimizer_name = "Adam"
        self.loss_name = "L1Loss"

        self.checkpoint = None

class TrainSettings(Settings):
    def __init__(self):
        super(TrainSettings, self).__init__()
        self.train_batch_size = 64
        self.eval_batch_size = 16

        self.hrWidth = 1920
        self.hrHeight = 1080

        self.epochs = 100
        self.learning_rate = 0.0001

        self.train_data_path = []
        self.eval_data_path = []

        self.hr_patch_size = 512

        self.weight_path = "checkpoint/"

class TestSettings(Settings):
    def __init__(self):
        super(TestSettings, self).__init__()

        self.test_data_path = []

        self.output_path = "result/"

def read_config(config_file : str)-> TestSettings:
    if not os.path.exists(config_file):
        print("Config file: ", config_file, " could not be found!")
        quit()

    with open(config_file, "r") as config:
        data = json.load(config)

    return data