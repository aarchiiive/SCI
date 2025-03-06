import os
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader

from darknet import DarkNet53, darknet53

device = torch.device('cuda:0')
model = darknet53().cuda()
model.eval().cuda()

# Dummy input
# input_size = (720, 1080)
# input_size = (832, 658) # Exdark
# input_size = (900, 1600) # nuImages
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