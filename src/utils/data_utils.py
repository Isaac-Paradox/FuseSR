import math
import numpy as np

HR_postfix_name = "_HR"
AA_postfix_name = "_AA"

reference_name = "Reference"
input_name = "Input"
raw_motion_depth_name = "RawMotionDepth"

motion_name = "Motion"
normal_name = "Normal"
depth_name = "Depth"

irradiance_name = "Irradiance"
precomputed_BRDF_name = "BRDF"

history_result_name = "History_Result"

data_channels = {}
data_channels[reference_name] = 3
data_channels[input_name] = 3
data_channels[motion_name] = 2

data_channels[normal_name] = 3
data_channels[depth_name] = 1

data_channels[irradiance_name] = 3
data_channels[precomputed_BRDF_name] = 3

class data_spliter:
    def __init__(self, upsample_factor : int, patch_size : int, hrWidth : int, hrHeight : int):
        self.upsample_factor = upsample_factor
        self.patch_size = patch_size

        self.hrWidth = hrWidth
        self.hrHeight = hrHeight
        self.lrWidth = int(self.hrWidth / self.upsample_factor)
        self.lrHeight = int(self.hrHeight / self.upsample_factor)

        self.u_num, self.v_num, self.patch_num = data_spliter.GetPatchNum(upsample_factor, patch_size, hrWidth, hrHeight)

        self.u_offset = self.patch_size - (self.u_num * self.patch_size - self.lrWidth) / (self.u_num - 1)
        self.v_offset = self.patch_size - (self.v_num * self.patch_size - self.lrHeight) / (self.v_num - 1)

        self.u_patch_motion_scale = (float(self.lrWidth)/ self.patch_size)
        self.v_patch_motion_scale = (float(self.lrHeight)/ self.patch_size)

    def GetPatchNum(upsample_factor : int, patch_size : int, hrWidth : int, hrHeight : int):
        u_num = math.ceil(hrWidth / patch_size)
        v_num = math.ceil(hrHeight / patch_size)

        return u_num, v_num, u_num * v_num

    def SplitDatas(self, datas : dict) -> dict:
        output = []
        for _ in range(self.patch_num):
            output.append({})
        for key, value in datas.items():
            splited_datas = self.SplitImages(value, HR_postfix_name in key or key == reference_name)
            for i in range(self.patch_num):
                splited_data = splited_datas[i]
                if key == motion_name:
                    splited_data = self.PatchMotionScale(splited_data)
                output[i][key] = splited_data
           
        return output

    def SplitData(self, datas : dict, index : int) -> dict:
        output = {}
        for key, value in datas.items():
            splited_data = self.SplitImage(value, HR_postfix_name in key or key == reference_name, index)
            if key == motion_name:
                splited_data = self.PatchMotionScale(splited_data)
            output[key] = splited_data
        return output

    def SplitImage(self, image : np.ndarray, is_upsampled : bool, index : int):
        u = index // self.u_num
        v = index % self.v_num
        u_start = int(u * self.u_offset)
        v_start = int(v * self.v_offset)
        if(is_upsampled):
            splited_img = image[:, v_start * self.upsample_factor:(v_start+self.patch_size) * self.upsample_factor,
                            u_start * self.upsample_factor:(u_start+self.patch_size) * self.upsample_factor].copy()
        else:
            splited_img = image[:, v_start:v_start+self.patch_size,u_start:u_start+self.patch_size].copy()

        return splited_img

    def SplitImages(self, image : np.ndarray, is_upsampled : bool):
        splited_imgs = list()   
        for u in range(self.u_num):
            for v in range(self.v_num):
                u_start = int(u * self.u_offset)
                v_start = int(v * self.v_offset)
                if(is_upsampled):
                    splited_img = image[:, v_start * self.upsample_factor:(v_start+self.patch_size) * self.upsample_factor,
                                    u_start * self.upsample_factor:(u_start+self.patch_size) * self.upsample_factor].copy()
                else:
                    splited_img = image[:, v_start:v_start+self.patch_size,u_start:u_start+self.patch_size].copy()
                splited_imgs.append(splited_img)

        return splited_imgs

    def PatchMotionScale(self, motion : np.ndarray):
        motion[0:1,:,:] = motion[0:1,:,:] * self.u_patch_motion_scale
        motion[1:2,:,:] = motion[1:2,:,:] * self.v_patch_motion_scale
        return motion