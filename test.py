
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import random
import torch
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from network.dataset import * 
import utils
from utils.settings import read_config

class FuseSRTestDataset(FuseSRDataset):
    def __init__(self, setting):
        super(FuseSRTestDataset, self).__init__(setting)

        for [data_path, selection, sel_info] in setting["test_data_path"]:
            if selection == "range":
                [start,end] = sel_info
                file_list = []
                for filename in os.listdir(data_path):
                    name, ext = os.path.splitext(filename)
                    if ext == ".npz":   
                        frame_index = int(name)
                        success, input_frame = self.add_valid_files(os.path.join(data_path, "{0}.npz"), frame_index)  
                        if success:
                            file_list.append(input_frame)
                file_list.sort(key = lambda x: x["frame_id"])
                self.input_files.extend(file_list[start:end])
            elif selection == "index":
                file_list = []
                for filename in os.listdir(data_path):
                    name, ext = os.path.splitext(filename)
                    if ext == ".npz":   
                        frame_index = int(name)
                        if frame_index in sel_info:
                            success, input_frame = self.add_valid_files(os.path.join(data_path, "{0}.npz"), frame_index)  
                            if success:
                                file_list.append(input_frame)
                file_list.sort(key = lambda x: x["frame_id"])
                self.input_files.extend(file_list)

def test(settings):
    random.seed(42)

    network_module = getattr(__import__("network.models", fromlist=[settings['model_file']]), settings['model_file'])
    net : torch.nn.Module = getattr(network_module, settings['model_name'])(settings)

    if 'checkpoint' not in settings or settings['checkpoint'] == '':
        ckpt_path = os.path.join(settings["weight_path"], settings["job_name"], settings["job_name"] + '.pt')
    else:
        ckpt_path = settings['checkpoint']

    if not os.path.exists(ckpt_path):
        print("checkpoint:%s not exist"%(ckpt_path))
        quit()
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['model_state'])
    net = net.cuda()
    net.eval()
    
    test_data = FuseSRTestDataset(settings)
    test_data_loader = DataLoader(test_data, 1, False)

    os.makedirs(os.path.join(settings['test_result'], settings['job_name'], "result"), exist_ok=True)

    with open(os.path.join(settings['test_result'], settings['job_name'], "result.txt"), mode='w', encoding='utf-8') as result_file:
        with torch.no_grad():
            total_ssim, total_psnr = 0., 0.
            for data, target in test_data_loader:
                frame_id = data["frame_id"][0].item()
                target = target.cuda()
                for k in data:
                    data[k] = data[k].cuda()
                output = net(data)
                ssim = utils.ssim(output, target).item()
                psnr = utils.hdr_psnr(output, target).item()
                output = output[0].detach().cpu().permute([1, 2, 0]).numpy()
                target = target[0].detach().cpu().permute([1, 2, 0]).numpy()
                cv2.imwrite(os.path.join(settings['test_result'], settings['job_name'], "result", "result_" + str(frame_id) + ".exr"), output[:,:,::-1])
                cv2.imwrite(os.path.join(settings['test_result'], settings['job_name'], "result", "truth_" + str(frame_id) + ".exr"), target[:,:,::-1])


                info = '[%d]%s SSIM: %.6f PSNR: %.6f' % (frame_id, settings['job_name'], ssim, psnr)

                print(info)
                result_file.write(info + '\n')

                total_ssim += ssim
                total_psnr += psnr

            data_num = len(test_data_loader)
            total_ssim /= data_num
            total_psnr /= data_num

            info = '%s AVG SSIM: %.6f PSNR: %.6f' % (settings['job_name'], total_ssim, total_psnr)
            print(info)
            result_file.write(info + '\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='', help='Which config to read')
    parser.add_argument('--check_point', type=str, default='', help='check point')
    parser.add_argument('-g', '--gpu_id', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    settings = read_config(args.config)

    if args.check_point != '': 
        settings['checkpoint'] = args.check_point

    test(settings)

    