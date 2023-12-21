import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torchvision
from typing import List, Callable, Any
from threading import Lock

perceptual_loss = None
ssim_loss = None
devices = None

def L2Loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.square(output - target).mean()


def sssr_loss(output: torch.Tensor, target: torch.Tensor, w1: float = 0.5, w2: float = 0.1):
    return F.l1_loss(output, target) + w1 * nsrr_loss(output, target, w2)


def nsrr_loss(output: torch.Tensor, target: torch.Tensor, w: float = 0.1) -> torch.Tensor:
    """
    Computes the loss as defined in the NSRR paper.
    """
    # loss_ssim = 1 - pytorch_ssim.ssim(output, target)
    loss_perception = 0
    global perceptual_loss, ssim_loss, devices
    if perceptual_loss is None:
        perceptual_loss = PerceptualLossManager()
        ssim_loss = utils.SSIM().cuda()
        # perceptual_loss.lom = perceptual_loss.lom.cuda()
        # if len(devices) > 1:
        #     print(devices)
        #     ssim_loss = nn.DataParallel(ssim_loss, devices).cuda()
        #     perceptual_loss.lom = nn.DataParallel(perceptual_loss.lom, devices).cuda()
    loss_ssim = 1 - ssim_loss(output, target)
    if len(devices) > 1:
        loss_ssim = loss_ssim.mean()
    # print(loss_ssim)
    conv_layers_output = perceptual_loss.get_vgg16_conv_layers_output(output)
    conv_layers_target = perceptual_loss.get_vgg16_conv_layers_output(target)
    # conv_layers_output = PerceptualLossManager().get_vgg16_conv_layers_output(output)
    # conv_layers_target = PerceptualLossManager().get_vgg16_conv_layers_output(target)
    for i in range(len(conv_layers_output)):
        loss_perception += feature_reconstruction_loss(conv_layers_output[i], conv_layers_target[i])
    loss = loss_ssim + w * loss_perception
    return loss

def feature_reconstruction_loss(conv_layer_output: torch.Tensor, conv_layer_target: torch.Tensor) -> torch.Tensor:
    """
    Computes Feature Reconstruction Loss as defined in Johnson et al. (2016)
    todo: syntax
    Justin Johnson, Alexandre Alahi, and Li Fei-Fei. 2016. Perceptual losses for real-time
    style transfer and super-resolution. In European Conference on Computer Vision.
    694–711.
    Takes the already-computed output from the VGG16 convolution layers.
    """
    if conv_layer_output.shape != conv_layer_target.shape:
        raise ValueError("Output and target tensors have different dimensions!")
    loss = conv_layer_output.dist(conv_layer_target, p=2) / torch.numel(conv_layer_output)
    return loss

class SingletonPattern(type):
    """
    see: https://refactoring.guru/fr/design-patterns/singleton/python/example
    """
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class PerceptualLossManager(metaclass=SingletonPattern):
    """
    Singleton
    """
    # Init
    def __init__(self):
        self.vgg_model = torchvision.models.vgg16(pretrained=True, progress=True)
        self.vgg_model.eval()
        """ 
            Feature Reconstruction Loss 
            - needs output from each convolution layer.
        """
        self.layer_predicate = lambda name, module: type(module) == nn.Conv2d
        
        self.lom = LayerOutputModelDecorator(self.vgg_model.features, self.layer_predicate).cuda()
        for param in self.lom.parameters():
            param.requires_grad = False

    def get_vgg16_conv_layers_output(self, x: torch.Tensor)-> List[torch.Tensor]:
        """
        Returns the list of output of x on the pre-trained VGG16 model for each convolution layer.
        """
        return self.lom(x)

class LayerOutputModelDecorator(nn.Module):
    """
    A Decorator for a Model to output the output from an arbitrary set of layers.
    """

    def __init__(self, model: nn.Module, layer_predicate: Callable[[str, nn.Module], bool]):
        super(LayerOutputModelDecorator, self).__init__()
        self.model = model
        self.layer_predicate = layer_predicate

        self.output_layers = []

        def _layer_forward_func(layer_index: int) -> Callable[[nn.Module, Any, Any], None]:
            def _layer_hook(module_: nn.Module, input_, output) -> None:
                self.output_layers[layer_index] = output
            return _layer_hook
        self.layer_forward_func = _layer_forward_func

        for name, module in self.model.named_children():
            if self.layer_predicate(name, module):
                module.register_forward_hook(
                    self.layer_forward_func(len(self.output_layers)))
                self.output_layers.append(torch.Tensor())

    def forward(self, x) -> List[torch.Tensor]:
        # with torch.no_grad():
        self.model(x)
        return self.output_layers