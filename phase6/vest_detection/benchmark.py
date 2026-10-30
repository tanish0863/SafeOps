import sys, os
sys.path.insert(0, os.path.abspath('src'))
import time, torch, json, numpy as np
import cv2
from vest_detector import VestDetector

model_path = os.path.abspath('models/yolov8n_vest_best.pt')
detector = VestDetector(model_path=model_path, device='cuda')

# Warmup
dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
for _ in range(10):
    _ = detector.detect(dummy)

# Full image benchmark (100 iterations)
latencies_cuda = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = detector.detect(dummy)
    latencies_cuda.append((time.perf_counter() - t0) * 1000.0)

# Torso inference benchmark (100 iterations)
torso_box = [400, 150, 700, 550]
latencies_torso = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = detector.assess_torso(dummy, torso_box)
    latencies_torso.append((time.perf_counter() - t0) * 1000.0)

vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
max_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

# CPU benchmark (20 iterations)
detector_cpu = VestDetector(model_path=model_path, device='cpu')
latencies_cpu = []
for _ in range(20):
    t0 = time.perf_counter()
    _ = detector_cpu.detect(dummy)
    latencies_cpu.append((time.perf_counter() - t0) * 1000.0)

metrics = {
    'device': torch.cuda.get_device_name(0),
    'cuda_latency_mean_ms': round(float(np.mean(latencies_cuda)), 2),
    'cuda_latency_median_ms': round(float(np.median(latencies_cuda)), 2),
    'cuda_latency_p95_ms': round(float(np.percentile(latencies_cuda, 95)), 2),
    'cuda_fps': round(1000.0 / float(np.mean(latencies_cuda)), 1),
    'torso_latency_mean_ms': round(float(np.mean(latencies_torso)), 2),
    'cpu_latency_mean_ms': round(float(np.mean(latencies_cpu)), 2),
    'cpu_fps': round(1000.0 / float(np.mean(latencies_cpu)), 1),
    'vram_allocated_mb': round(vram_allocated_mb, 2),
    'vram_reserved_mb': round(vram_reserved_mb, 2),
    'vram_peak_allocated_mb': round(max_vram_mb, 2)
}

print('BENCHMARK_RESULTS:', json.dumps(metrics, indent=2))

os.makedirs('artifacts/evaluation', exist_ok=True)
with open('artifacts/evaluation/latency_benchmark.json', 'w') as f:
    json.dump(metrics, f, indent=2)
