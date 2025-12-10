import numpy as np
import time
import os
import sys
import threading
import torch
import torch.nn as nn
from tqdm import tqdm

# 모델 import
try:
    from ipconv.models import (
        MaskedRCNN_ViT_B_FPN_Contexted, MaskedRCNN_ViT_L_FPN_Contexted, MaskedRCNN_ViT_H_FPN_Contexted,
        CascadeMaskRCNN_Swin_B_Contexted, CascadeMaskRCNN_Swin_L_Contexted,
    )
except ImportError:
    print("[Warning] ipconv.models not found. Please check your python path.")
    sys.exit(1)

# --- 1. PowerEstimator (power_live.py와 동일 로직) ---
class PowerEstimator:
    def __init__(self, idle_load_duration=3, idle_load_samples=10, sampling_rate=30):
        self.rail_paths = [] 
        
        # power_live.py와 똑같이 hwmon4의 in1(VDD_IN)을 우선적으로 찾음
        base_path = '/sys/class/hwmon/hwmon4/'
        
        print(f"[PowerEstimator] Scanning sensors in {base_path}...")
        
        # 1. 명시적으로 1번 채널(VDD_IN) 확인
        v1 = os.path.join(base_path, "in1_input")
        c1 = os.path.join(base_path, "curr1_input")
        l1 = os.path.join(base_path, "in1_label")
        
        target_found = False

        # 라벨 확인 (VDD_IN인지)
        if os.path.exists(l1):
            try:
                with open(l1, 'r') as f:
                    label = f.read().strip()
                if "VDD_IN" in label:
                    self.rail_paths.append((v1, c1))
                    print(f"  - [TARGET] Found Main Power Rail 1 ({label}) -> Using this.")
                    target_found = True
            except:
                pass
        
        # 만약 라벨 확인이 안 되더라도, power_live.py처럼 그냥 1번 채널 강제 사용 (Fallback)
        if not target_found:
            if os.path.exists(v1) and os.path.exists(c1):
                self.rail_paths.append((v1, c1))
                print("  - [FALLBACK] Label check failed, but using Rail 1 (Same as power_live.py).")
            else:
                print("[ERROR] Sensor files not found! Check connection.")

        self.idle_load = 0
        self._estimate_standby_load(idle_load_duration, idle_load_samples)
        self.sampling_interval = 1.0 / sampling_rate

    def _get_instant_power(self):
        total_power_mw = 0
        # power_live.py와 동일한 단순 파일 읽기 방식
        for v_path, c_path in self.rail_paths:
            try:
                with open(v_path, "r") as fv:
                    voltage_mv = float(fv.read()) # mV
                with open(c_path, "r") as fc:
                    current_ma = float(fc.read()) # mA
                
                # P(mW) = V(mV) * I(mA) / 1000
                total_power_mw += (voltage_mv * current_ma) / 1000.0
            except:
                pass 
        return total_power_mw

    def _estimate_standby_load(self, duration, samples=10.0):
        print(f"[Power] Estimating idle power for {duration}s...")
        idle_load = 0.0
        for i in range(int(samples)):
            idle_load += self._get_instant_power()
            time.sleep(duration / float(samples))
        
        if samples > 0:
            self.idle_load = idle_load / samples
        print(f"[Power] Idle power: {self.idle_load:.3f} mW")

    def estimate_fn_power(self, fn):
        start = time.time()
        total_energy_mj = 0
        total_energy_over_idle_mj = 0
        
        th = threading.Thread(target=fn)
        th.start()

        while True:
            time.sleep(self.sampling_interval)
            poll_start_time = time.time()
            
            if th.is_alive():
                current_power = self._get_instant_power()
                poll_time = time.time() - poll_start_time
                
                # 에너지(mJ) = 전력(mW) * 시간(s)
                interval = self.sampling_interval + poll_time
                total_energy_mj += current_power * interval
                
                # Dynamic Energy 적분
                over_idle = max(0, current_power - self.idle_load)
                total_energy_over_idle_mj += over_idle * interval
            else:
                break
                
        total_time = time.time() - start
        th.join()
        return total_energy_mj, total_energy_over_idle_mj, total_time

# --- 2. 측정 함수 ---
def measure_latency_memory_energy(
    model: nn.Module,
    patch_keep_rate: float,
    method: str,
    input_size: int,
    p_est: PowerEstimator
):
    model.eval()
    dummy_input = np.zeros((input_size, input_size, 3))
    block_size = 16
    num_blocks_sqrt = input_size // block_size

    dmap = torch.zeros((1, num_blocks_sqrt, num_blocks_sqrt, 1), dtype=torch.float32, device="cuda")
    num_patches = num_blocks_sqrt * num_blocks_sqrt
    num_keep = int(num_patches * patch_keep_rate)
    
    if num_keep > 0:
        idx_rand = torch.randperm(num_patches)[:num_keep]
        dmap.view(-1)[idx_rand] = 1.0

    # [수정] 반복 횟수 5 -> 50으로 증가 (짧으면 측정 오차 큼)
    num_warmup = 3
    num_repeats = 5

    if method == "vanilla": inference_func = model.forward
    elif method == "ours": inference_func = model.forward_contexted
    elif method == "eventful": inference_func = model.forward_eventful
    elif method == "maskvd": inference_func = model.forward_maskvd
    elif method == "stgt": inference_func = model.forward_stgt
    else: raise ValueError(f"Unknown method: {method}")

    # Warmup
    for _ in range(num_warmup):
        output = inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True)
    
    cache_args = {"anchor_features": output[1]} if method != "vanilla" else {}

    # FLOPs
    model.counting()
    model.clear_counts()
    inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True, **cache_args)
    counts = model.total_counts()
    model.clear_counts()

    # Workload
    def workload():
        torch.cuda.synchronize()
        for _ in range(num_repeats):
            inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True, **cache_args)
        torch.cuda.synchronize()

    # 측정
    total_energy_mj, total_energy_over_idle_mj, total_time_sec = p_est.estimate_fn_power(workload)

    # [수정] 에너지 계산 수식 정정
    # Energy(J) = Total mJ / 1000 / 횟수
    energy_per_inference_J = (total_energy_mj / 1000.0) / num_repeats
    
    # Dynamic Energy(J) = Dynamic mJ / 1000 / 횟수 (뺄셈할 필요 없음, 이미 적분됨)
    dynamic_energy_per_inference_J = (total_energy_over_idle_mj / 1000.0) / num_repeats
    
    latency_sec = total_time_sec / num_repeats
    
    cache_size = 0
    if method != "vanilla":
        cache = output[1]
        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                cache_size += value.element_size() * value.numel()

    return latency_sec, energy_per_inference_J, dynamic_energy_per_inference_J, cache_size, counts


# --- 3. 메인 ---
@torch.no_grad()
def main():
    print("Initializing Power Estimator...")
    # PowerEstimator를 모델 로드 전에 초기화해서 순수 Idle(5W)을 잡도록 유도
    p_est = PowerEstimator(idle_load_duration=3, sampling_rate=20) 

    models_dict = {
        # "ViT-base": MaskedRCNN_ViT_B_FPN_Contexted,
        "ViT-large": MaskedRCNN_ViT_L_FPN_Contexted,
        # "Swin-base": CascadeMaskRCNN_Swin_B_Contexted,
    }

    # keep_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    keep_rates = [1.0]
    # methods = ["ours", "eventful", "maskvd", "stgt"]
    methods = ["vanilla"]
    input_sizes = [1024]
    
    print("\nStarting Measurement...")
    print("InputSize,Model,Method,KeepRate,Latency(s),TotalEnergy(J),DynEnergy(J),FLOPs,Cache(MB)")

    for input_size in input_sizes:
        for mname, model_class in models_dict.items():
            try:
                # 모델 로드 (여기서부터 전력이 12W로 뛸 수 있음 -> 정상적인 현상임)
                model = model_class("cuda")
                model.eval()
            except Exception as e:
                print(f"[Error] Failed to load model {mname}: {e}")
                continue

            for method in methods:
                for keep_rate in keep_rates:
                    try:
                        lat, energy, dyn_energy, cache, num_count = measure_latency_memory_energy(
                            model, keep_rate, method, input_size, p_est
                        )
                        flops = sum(num_count.values())
                        cache_mb = cache / (1024 * 1024)
                        
                        print(f"{input_size},{mname},{method},{keep_rate},{lat:.4f},{energy:.4f},{dyn_energy:.4f},{flops},{cache_mb:.2f}")
                        sys.stdout.flush()
                        
                    except Exception as e:
                        print(f"Error in {method} with {keep_rate}: {e}")

            del model
            torch.cuda.empty_cache()
            import gc
            gc.collect()
if __name__ == "__main__":
    main()