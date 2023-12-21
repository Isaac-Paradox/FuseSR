from ast import Str
from torch.utils.tensorboard import SummaryWriter
import os
import time

class Logger:
    def __init__(self, settings, force_print_log = False):
        self.force_print_log = force_print_log

        self.log_path = os.path.join(settings["log_folder"], settings["job_name"], time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.writers = {}

    def log_scalar(self, tag, scalar_value, global_step, component_name : Str = "" , print_log = False):
        self.get_writer(component_name).add_scalar(tag, scalar_value, global_step)
        if print_log or self.force_print_log:
            print('[%d] %s %s: %.6f' %(global_step + 1, component_name, tag, scalar_value))

    def get_writer(self, component_name : Str = "") -> SummaryWriter:
        if component_name == "":
            return self.writer
        else:
            if not component_name in self.writers:
                path = os.path.join(self.log_path, component_name)
                os.makedirs(path)
                self.writers[component_name] = SummaryWriter(path)
            return self.writers[component_name]