from torch.utils.data import Dataset
import os
import utils
import numpy as np
import torch

class FuseSRDataset(Dataset):
    def __init__(self, setting : utils.Settings):
        super(FuseSRDataset, self).__init__()

        self.check_data = setting["check_data"]
        self.model_input = {}

        for input_name in setting["model_input"]:
            is_history_information = "history_" in input_name
            input_keyword = input_name
            frame_offset = 0
            if is_history_information:
                input_keyword = input_keyword[8:]
                if input_keyword[0].isdigit():
                    frame_offset = int(input_keyword[0])
                    input_keyword = input_keyword[2:]
            if input_keyword == "HR_result":
                input_keyword = "Reference"

            if not frame_offset in self.model_input.keys():
                self.model_input[frame_offset] = {}
            self.model_input[frame_offset][input_name] = input_keyword
        self.model_input[0]["reference"] = "Reference"
        self.input_files = []
            
    def add_valid_files(self, path : str, frame_index : int):
        input_frame = {}
        input_frame["frame_id"] = frame_index
        for frame_offset, input_keywords in self.model_input.items():
            file_path = path.format(frame_index - frame_offset)
            if self.check_data and not os.path.exists(file_path):
                return False, input_frame
            input_frame[file_path] = input_keywords
        return True, input_frame

    def __getitem__(self, index):
        paths = self.input_files[index]
        item = {}
        try:
            for key, value in paths.items():
                if key == "frame_id" or key == "patch_id":
                    item[key] = torch.tensor(value)
                else:
                    data = np.load(key)
                    for input_name, input_keyword in value.items():
                        if input_name =="reference":
                            reference = data[input_keyword]
                        else:
                            item[input_name] = data[input_keyword]
        except Exception as e:
            print("error occur in loading file:%s"%(value))
            raise e
        return item, reference

    def __len__(self) -> int:
        return len(self.input_files)