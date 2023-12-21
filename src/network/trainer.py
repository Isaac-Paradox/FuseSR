import os

import torch
from torch.utils.data import DataLoader

from network.logger import Logger
import network.loss as loss_manager
from network.dataset import * 
import utils

import time
from tqdm import tqdm

class FuseSRPatchedTrainDataset(FuseSRDataset):
    def __init__(self, setting, data_paths, patch_num):
        super(FuseSRPatchedTrainDataset, self).__init__(setting)
        self.patch_num = patch_num

        for [data_path, [start,end]] in data_paths:
            self.data_path = data_path
            file_list = []
            for filename in os.listdir(data_path):
                path = os.path.normpath(filename)
                frame_index = int(path.split(os.sep)[-1])
                for i in range(self.patch_num):
                    success, input_frame = self.add_valid_files(os.path.join(self.data_path, "{0}", str(i) + ".npz"), frame_index)
                    if success:
                        input_frame["patch_id"] = i
                        file_list.append(input_frame)
            file_list.sort(key = lambda x: x["frame_id"])
            self.input_files.extend(file_list[start : end])

class FuseSRTrainDataset(FuseSRDataset):
    def __init__(self, setting, data_paths):
        super(FuseSRTrainDataset, self).__init__(setting)
        for [data_path, [start,end]] in data_paths:
            self.data_path = data_path
            file_list = []
            for filename in os.listdir(data_path):
                name, ext = os.path.splitext(filename)
                if ext == ".npz":
                    frame_index = int(name)
                    success, input_frame = self.add_valid_files(os.path.join(self.data_path, "{0}.npz"), frame_index)  
                    if success:
                        file_list.append(input_frame)
            file_list.sort(key = lambda x: x["frame_id"])
            self.input_files.extend(file_list[start:end])


class Trainer:
    def __init__(self, settings):
        self.muti_gpu = len(settings["gpu"]) > 1 
        self.epochs = settings["epochs"]
        self.settings = settings

        self.loss_func = Trainer.get_loss_function(settings)

        self.job_name = self.settings["job_name"]

        self.ckpt_folder = os.path.join(self.settings["weight_path"], self.job_name)

        os.makedirs(self.ckpt_folder, exist_ok=True)
        os.makedirs(os.path.join("tmp", self.job_name), exist_ok=True)

        self.load_dataset()

        self.load_model()

        self.logger = Logger(settings)
        
    def load_model(self):
        network_module = getattr(__import__("network.models", fromlist=[self.settings["model_file"]]), self.settings["model_file"])
        
        if "checkpoint" not in self.settings:
            self.epoch = 0
            self.best_loss = 0
            self.converge_count = 0
            self.net : torch.nn.Module = getattr(network_module, self.settings["model_name"])(self.settings)
            if(self.muti_gpu):
                loss_manager.devices = self.settings["gpu"]
                self.net = torch.nn.DataParallel(self.net, self.settings["gpu"])
            torch.cuda.set_device(self.settings["gpu"][0])
            self.net = self.net.cuda()

            self.optimizer : torch.optim.Optimizer = getattr(torch.optim, self.settings["optimizer_name"])(self.net.parameters() ,lr = self.settings["learning_rate"])
        else:
            ckpt_path = os.path.join(self.settings["weight_path"], self.settings["checkpoint"])
            if not os.path.exists(ckpt_path):
                print("checkpoint:%s not exist"%(ckpt_path))
                quit()
            checkpoint = torch.load(ckpt_path)

            self.net : torch.nn.Module = getattr(network_module, self.settings["model_name"])(self.settings)
            self.net.load_state_dict(checkpoint['model_state'])
            if(self.muti_gpu):
                loss_manager.devices = self.settings["gpu"]
                self.net = torch.nn.DataParallel(self.net, self.settings["gpu"])
            torch.cuda.set_device(self.settings["gpu"][0])
            self.net = self.net.cuda()   

            self.optimizer : torch.optim.Optimizer = getattr(torch.optim, self.settings["optimizer_name"])(self.net.parameters() ,lr = self.settings["learning_rate"])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            self.epoch = checkpoint['epoch'] + 1
            if 'best_loss' in checkpoint:
                self.best_loss = checkpoint['best_loss']
                self.converge_count = checkpoint['converge_count']
                self.best_checkpoint = checkpoint
            else:
                self.best_loss = 0
                self.converge_count = 0
    
    def load_dataset(self):
        eval_num_worker = self.settings["num_workers"]
        if "patch_training" not in self.settings or self.settings["patch_training"]:
            _, _, patch_num = utils.data_spliter.GetPatchNum(self.settings["upsample_factor"], self.settings["hr_patch_size"], self.settings["hrWidth"], self.settings["hrHeight"])
            self.train_data = FuseSRPatchedTrainDataset(self.settings, self.settings["train_data_path"], patch_num)
            eval_num_worker //= 2
        else:
            self.train_data = FuseSRTrainDataset(self.settings, self.settings["train_data_path"])
        
        self.eval_data = FuseSRTrainDataset(self.settings, self.settings["eval_data_path"])

        self.train_loader = DataLoader(self.train_data, self.settings["train_batch_size"], True, num_workers=self.settings["num_workers"])
        self.train_data_size = len(self.train_loader)
        
        self.eval_loader = DataLoader(self.eval_data, self.settings["eval_batch_size"], True, num_workers=eval_num_worker)
        self.eval_data_size = len(self.eval_loader)

    def train(self):
        for epoch in range(self.epoch, self.epochs):
            train_loss = 0.0
            train_ssim = 0.0
            train_psnr = 0.0
            self.net.train()
            train_start = time.time()
            for data, target in tqdm(self.train_loader, "Training %s epoch[%d]"%(self.job_name, epoch)):
                target = target.cuda()
                for k in data:
                    data[k] = data[k].cuda()
                self.optimizer.zero_grad()
                outputs = self.net(data)
                loss : torch.Tensor = self.loss_func(outputs, target)
                loss.backward() 
                self.optimizer.step()
                with torch.no_grad():
                    # train_ssim += pytorch_ssim.ssim(outputs, target).item()
                    if loss_manager.ssim_loss is None:
                        loss_manager.ssim_loss = utils.SSIM().cuda()
                    train_ssim += loss_manager.ssim_loss(outputs, target).item()
                    train_psnr += utils.hdr_psnr(outputs, target).item()
                    train_loss += loss.item()
            train_end = time.time()
            train_loss /= self.train_data_size
            train_ssim /= self.train_data_size
            train_psnr /= self.train_data_size
            self.logger.log_scalar("train loss", train_loss, epoch)
            self.logger.log_scalar("train ssim", train_ssim, epoch)
            self.logger.log_scalar("train psnr", train_psnr, epoch)
            temp_checkpoint = {'epoch' : epoch,
                        'model_state' : self.net.state_dict() if not self.muti_gpu else self.net.module.state_dict(),
                        'optimizer_state' : self.optimizer.state_dict(),
                        'best_loss' : self.best_loss,
                        'converge_count' : self.converge_count}
            ckpt_path = os.path.join(self.ckpt_folder, self.job_name + '_temp.pt')
            print("Saving checkpoint to: " + ckpt_path)
            torch.save(temp_checkpoint, ckpt_path)

            # evaluation--------------------------------
            self.net.eval()
            # eval_loss = 0.
            eval_ssim = 0.
            eval_psnr = 0.
            eval_start = time.time()
            with torch.no_grad():
                for data, target in tqdm(self.eval_loader, "Evaluating %s epoch[%d]"%(self.job_name, epoch)):
                    torch.cuda.empty_cache()
                    target = target.cuda()
                    for k in data:
                        data[k] = data[k].cuda()
                    with torch.cuda.amp.autocast(False):
                        outputs = self.net(data)
                        # loss = self.loss_func(outputs, target)
                        # eval_loss += loss.item()
                    # eval_ssim += pytorch_ssim.ssim(outputs.float(), target.float()).item()
                    eval_ssim += loss_manager.ssim_loss(outputs.float(), target.float()).item()
                    eval_psnr += utils.hdr_psnr(outputs.float(), target.float()).item()
            # imageio.imwrite(os.path.join("tmp", self.job_name, "output.exr"), outputs.float().detach().cpu()[0].permute(1,2,0))
            # imageio.imwrite(os.path.join("tmp", self.job_name, "refernece.exr"), target.float().detach().cpu()[0].permute(1,2,0))
            eval_end = time.time()
            # eval_loss /= self.eval_data_size
            eval_ssim /= self.eval_data_size
            eval_psnr /= self.eval_data_size
            # self.logger.log_scalar("eval loss", eval_loss, epoch)
            self.logger.log_scalar("eval ssim", eval_ssim, epoch)
            self.logger.log_scalar("eval psnr", eval_psnr, epoch)
            
            if eval_psnr > self.best_loss:
                self.best_loss = eval_psnr
                self.best_checkpoint = {'epoch' : epoch,
                        'model_state' : self.net.state_dict() if not self.muti_gpu else self.net.module.state_dict(),
                        'optimizer_state' : self.optimizer.state_dict()}
                self.converge_count = 0
            else:
                self.converge_count += 1
            if self.converge_count == 5 or (epoch + 1 == self.epochs and self.converge_count < 5):
                ckpt_path = os.path.join(self.ckpt_folder, self.job_name + '_%d.pt'% (self.best_checkpoint['epoch'] + 1))
                print("Saving checkpoint to: " + ckpt_path)
                torch.save(self.best_checkpoint, ckpt_path)

            print('[%d] train loss: %.6f , train cost: %.6f s, eval psnr: %.6f , eval cost: %.6f s, total cost: %.6f s' 
            %(epoch + 1, train_loss, train_end - train_start, eval_psnr, eval_end - eval_start,  time.time() - train_start))

        ckpt_path = os.path.join(self.ckpt_folder, self.job_name + '.pt')
        print("Saving checkpoint to: " + ckpt_path)
        torch.save(self.best_checkpoint, ckpt_path)
        print('best evaluation loss: %.6f , in epoch %d'%(self.best_loss, (self.best_checkpoint['epoch'] + 1)))

    @classmethod
    def get_loss_function(cls, settings):
        loss_functions = [
            "nsrr_loss",
            "sssr_loss"
        ]
        if settings["loss_name"] in loss_functions:
            return getattr(__import__("network.loss", fromlist=[settings["loss_name"]]), settings["loss_name"])
        else:
            return getattr(torch.nn, settings["loss_name"])()