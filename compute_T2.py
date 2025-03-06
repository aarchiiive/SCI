import os
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')
    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


device = torch.device('cuda:0')
model = DecomNet().cuda()
model.eval().cuda()

# Dummy input
# input_size = (720, 1080)
# input_size = (832, 658) # Exdark
# input_size = (900, 1600) # nuImages
# input_size = (640, 640) # nuImages
input_size = (608, 608) # nuImages
# dummy_input = torch.randn(1, 3, 640, 640).to(device)

dummy_input = torch.randn(1, 3, *input_size).cuda()

with torch.no_grad():
    # GPU warm-up (10회 실행)
    for _ in range(100):
        _ = model(dummy_input)

    # 100번의 추론 실행 후 시간 측정
    times = []
    for _ in range(1000):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = model(dummy_input)
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

# 평균 실행 시간 계산 및 출력
avg_time = sum(times) / len(times)
print("Average inference time over 100 runs: {:.3f} ms".format(avg_time))