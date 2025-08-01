#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, aggregate=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    ssim_map = _ssim(img1, img2, window, window_size, channel)

    if aggregate == False:
        return ssim_map

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


def compute_gradient(image):
    """ 计算梯度图像（用于梯度损失） """
    if image.dim() == 3:  
        image = image.unsqueeze(0)  # shape: [1, C, H, W]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    # 复制 Sobel 核以匹配输入通道数
    sobel_x = sobel_x.repeat(image.shape[1], 1, 1, 1)
    sobel_y = sobel_y.repeat(image.shape[1], 1, 1, 1)

    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1])

    return torch.abs(grad_x) + torch.abs(grad_y)

def update_lambda_schedule(iteration, total_iter, opt, args):
    progress = iteration / total_iter
    opt.lambda_dssim = args.lambda_start - args.lambda_end * (1 / (1 + math.exp(-opt.k * (progress - opt.p))))


from lpipsPyTorch import lpips

def compute_robust_loss(image, gaussians, visibility_filter, gt_image, iteration, opt, args):
    opacity = gaussians.get_opacity
    opacity = opacity.clamp(1e-6, 1-1e-6)
    log_opacity = opacity * torch.log(opacity)
    log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
    sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()

    color_diff = (image - gt_image).abs().mean(dim=0)
    
    # 动态权重调度
    t = iteration / opt.iterations
    beta = args.beta
    w_consist = beta * t  # 线性增加到--0.2
    gama = args.gama
    w_stability = gama * (1 - t)  # 逐渐降低--0.05
    # 损失组合
    total_loss = (
        w_consist * color_diff.mean() +
        w_stability * sparse_loss
    )
    
    return total_loss

def compute_color_variance(gaussians, viewpoint_cam):
    """基于预计算的多视角颜色方差"""
    # 在GaussianModel类中添加属性存储颜色方差
    if not hasattr(gaussians, 'color_variance'):
        # 初始化时计算所有点的颜色方差
        gaussians.color_variance = torch.zeros(gaussians.get_xyz.shape[0], device='cuda')
    
    # 仅更新可见点的方差（减少计算量）
    colors = gaussians.get_color(viewpoint_cam)
    gaussians.color_variance = torch.var(colors, dim=1).mean(dim=1)
    
    return gaussians.color_variance