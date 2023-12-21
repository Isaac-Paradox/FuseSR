import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "src"))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import utils
from utils.data_utils import *

ext = ".exr"
from postprocrss_utils import *

upsample_scales = [
    4, 8
]

class DataPacking:
    def __init__(self, root_dir: str, out_dir: str, hrWidth : int, hrHeight : int, 
                 patch_size : int, max_workers=1):
        self.root_dir = root_dir
        self.out_dir = out_dir

        self.package_input_datas = {
            input_name : input_name,
            normal_name : normal_name,
            raw_motion_depth_name : raw_motion_depth_name,
            precomputed_BRDF_name : precomputed_BRDF_name,
            precomputed_BRDF_name + HR_postfix_name : os.path.join("HR", precomputed_BRDF_name),
            reference_name : os.path.join("AA", reference_name)
        }

        self.data_spliters = []
        for factor in upsample_scales:
            self.data_spliters.append(data_spliter(factor, patch_size, hrWidth, hrHeight))

        self.max_workers = max_workers if max_workers > 0 else None

        self.exposure = 1
        self.max = 10
        self.min = 0

        self.save_preview = False
        self.delete_source_file = True
        self.force = False
            
    def PackData(self):
        os.makedirs(self.out_dir, exist_ok=True)
        
        frame_indices = sorted(map(lambda filename: int(os.path.splitext(filename)[0]), os.listdir(os.path.join(self.root_dir, 'AA', 'Reference'))))
        if self.max_workers == 1:
            for frame_index in tqdm(frame_indices):
                enabled = self.force or not os.path.exists(os.path.join(self.out_dir, str(frame_index) + ".npz"))
                self.PackFrame(frame_index, enabled)
        else:
            enables = [(self.force or not os.path.exists(os.path.join(self.out_dir, str(frame_index) + ".npz"))) for frame_index in frame_indices]
            process_map(self.PackFrame, frame_indices, enables, max_workers=self.max_workers, chunksize=10)

    def PackFrame(self, frame_index : int, enabled=True):
        if not enabled: return

        for i, factor in enumerate(upsample_scales):
            data_path = os.path.join(self.out_dir, str(factor) + 'x')
            train_data_path = os.path.join(data_path, "train")
            os.makedirs(train_data_path, exist_ok=True)

            input_data = {}
            for data_name, path_name in self.package_input_datas.items():
                if data_name == reference_name or AA_postfix_name in data_name or HR_postfix_name in data_name:
                    path = os.path.join(self.root_dir, path_name, str(frame_index) + ext)
                else:
                    path = os.path.join(self.root_dir, str(factor) + 'x', data_name, str(frame_index) + ext)
                data = DataPacking.loadHdr(path)

                if self.delete_source_file:
                    os.remove(path)
                input_data[data_name] = data

            data_bundle = {}

            LR_input = self.ExpoNClip(input_data[input_name])
            data_bundle[normal_name] = input_data[normal_name]
            motion, depth = self.GetMotionDepth(input_data[raw_motion_depth_name])
            data_bundle[motion_name] = motion
            data_bundle[depth_name] = depth

            BRDF = self.MapBRDF(input_data[precomputed_BRDF_name])
            data_bundle[irradiance_name] = self.GetIrradiance(LR_input, BRDF)
            data_bundle[precomputed_BRDF_name + HR_postfix_name] = self.MapBRDF(input_data[precomputed_BRDF_name + HR_postfix_name])

            data_bundle[reference_name] = self.ExpoNClip(input_data[reference_name])
            
            np.savez_compressed(os.path.join(data_path, str(frame_index) + ".npz"), **data_bundle)
            frame_path = os.path.join(train_data_path, str(frame_index))
            os.makedirs(frame_path, exist_ok=True)

            splited_datas = self.data_spliters[i].SplitDatas(data_bundle)

            for j in range(self.data_spliters[i].patch_num):
                if np.sum(splited_datas[j][precomputed_BRDF_name + HR_postfix_name]) < 1e-7:
                    continue
                np.savez_compressed(os.path.join(frame_path, str(j) + ".npz"), **(splited_datas[i]))


    def ExpoNClip(self, x : np.ndarray):
        return np.clip(x * self.exposure, self.min, self.max)
    
    def loadHdr(imName):
        im = cv2.imread(imName, -1)
        if im is None:
            raise ValueError(f"ERROR: Open {imName} failed!")
        # if len(im.shape) == 3:
        return np.transpose(im[:, :, 0:3][:, :, ::-1], [2, 0, 1]).copy()
        # else:
        #     return np.expand_dims(im, 0).copy()

    def GetMotionDepth(self, raw_motion_depth : np.ndarray):
        motion = raw_motion_depth[0:2,:,:]
        motion[0:1,:,:] = motion[0:1,:,:] * -1.0
        depth = raw_motion_depth[2:3,:,:]
        return motion, depth

    def MapBRDF(self, BRDF : np.ndarray) -> np.ndarray:
        # rewrite in numpy
        return F.softplus(torch.tensor(BRDF), 100).numpy()

    def GetIrradiance(self, color : np.ndarray, BRDF : np.ndarray) -> np.ndarray:
        return np.where(BRDF == 0, BRDF, color / BRDF)

def main():
    parser = argparse.ArgumentParser(prog="data Post Process", description="data Post Process program")
    parser.add_argument("-i", "--input", required=True, type=str, help="input data path", metavar="input_path")
    parser.add_argument("-o", "--output", required=True, type=str, help="output data path", metavar="output_path")
    parser.add_argument("-W", "--hr_width", required=True, type=int, help="high resolution width", metavar="high_resolution_width")
    parser.add_argument("-H", "--hr_height", required=True, type=int, help="high resolution height", metavar="high_resolution_height")
    parser.add_argument("-s", "--patch_size", default = 128, type=int, help="patch size", metavar="patch_size")
    parser.add_argument("-p", "--preview", action="store_true", help="save preview file")
    parser.add_argument("-d", "--delete", action="store_true", help="delete source file")
    parser.add_argument("--max_workers", default=1, type=int, help="Number of workers")
    parser.add_argument("-f", "--force", action="store_true", help="force pack")
    parser.add_argument("-e", "--exposure", default=1, type=float, help="Exposure")
    parser.add_argument("--max", default=10, type=float, help="max value")
    parser.add_argument("--min", default=0, type=float, help="min value")
    args = vars(parser.parse_args())

    packing_tool = DataPacking(args["input"], args["output"], args["hr_width"], args["hr_height"], args["patch_size"], args["max_workers"])
    packing_tool.save_preview = args["preview"]
    packing_tool.delete_source_file = args["delete"]
    packing_tool.force = args["force"]
    packing_tool.max = args["max"]
    packing_tool.min = args["min"]
    packing_tool.exposure = args["exposure"]
    packing_tool.PackData()

if __name__ == "__main__":
    main()
