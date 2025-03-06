import torch
from model import Finetunemodel

# 모델 로드 및 GPU 이동
model = Finetunemodel('weights/easy.pt').cuda()

# 입력 데이터 생성 (배치 크기 1, 채널 3, 720x1080 이미지)
dummy_input = torch.randn(1, 3, 720, 1080).cuda()

# GPU warm-up (초기 오버헤드를 줄이기 위해 10회 실행)
for _ in range(10):
    _ = model(dummy_input)

# 100번의 추론 실행 후 시간 측정
times = []
for _ in range(100):
    with torch.no_grad():
        torch.cuda.synchronize()  # GPU 연산 동기화
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = model(dummy_input)
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # 실행 시간(ms) 저장

# 평균 실행 시간 계산 및 출력
avg_time = sum(times) / len(times)
print("Average inference time over 100 runs: {:.3f} ms".format(avg_time))
