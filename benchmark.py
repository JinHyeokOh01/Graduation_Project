import time
import subprocess
import numpy as np
from PIL import Image
import tensorflow as tf
from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                            ConfigureParams, InputVStreamParams, 
                            OutputVStreamParams, InferVStreams, FormatType)
from pathlib import Path

FULL_TFLITE = "mobilenet_v1_full.tflite"
FULL_HEF = "mobilenet_v1_full_v3.hef"
NPU_HEF = "npu_blocks_1_8_float32.hef"
CPU_UINT8 = "cpu_blocks_9_end_uint8.tflite"

TEST_IMAGES = []
test_dir = Path("test_images")
labels = {'cat': 281, 'dog': 207, 'car': 817}

for img_path in sorted(test_dir.glob("*.jpg")):
    for key, val in labels.items():
        if img_path.stem.lower().startswith(key):
            TEST_IMAGES.append({"path": str(img_path), "label": val, "name": img_path.stem})
            break

print(f"로드된 이미지: {len(TEST_IMAGES)}장\n")

WARMUP = 3
RUNS = 10

# ==================== 에너지 측정 함수 ====================
def get_cpu_temp():
    """CPU 온도 (°C)"""
    try:
        result = subprocess.run(['vcgencmd', 'measure_temp'], 
                              capture_output=True, text=True, timeout=1)
        temp_str = result.stdout.strip()
        return float(temp_str.replace("temp=", "").replace("'C", ""))
    except:
        return None

def get_cpu_freq():
    """CPU 주파수 (MHz)"""
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
            return int(f.read().strip()) / 1000
    except:
        return None

# ==================== 기존 함수들 ====================
def preprocess(path):
    img = Image.open(path).convert('RGB').resize((224, 224))
    return np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

def run_cpu_baseline(data):
    interp = tf.lite.Interpreter(FULL_TFLITE)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    
    data_float = (data.astype(np.float32) / 127.5) - 1.0
    
    for _ in range(WARMUP):
        interp.set_tensor(inp[0]['index'], data_float)
        interp.invoke()
    
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        interp.set_tensor(inp[0]['index'], data_float)
        interp.invoke()
        times.append((time.perf_counter() - start) * 1000)
    
    pred = interp.get_tensor(out[0]['index'])[0]
    return np.mean(times), pred

def run_npu_full(data):
    times = []
    pred = None
    
    with VDevice() as vdev:
        hef = HEF(FULL_HEF)
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = vdev.configure(hef, cfg)[0]
        
        with ng.activate():
            inp_params = InputVStreamParams.make_from_network_group(ng, quantized=True, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make_from_network_group(ng, quantized=True, format_type=FormatType.UINT8)
            
            with InferVStreams(ng, inp_params, out_params) as infer:
                hef_info = HEF(FULL_HEF)
                inp_name = hef_info.get_input_vstream_infos()[0].name
                out_name = hef_info.get_output_vstream_infos()[0].name
                
                for _ in range(WARMUP):
                    infer.infer({inp_name: data})
                
                for _ in range(RUNS):
                    start = time.perf_counter()
                    result = infer.infer({inp_name: data})
                    times.append((time.perf_counter() - start) * 1000)
                
                out_uint8 = result[out_name]
                pred = out_uint8.astype(np.float32) / 255.0
                if len(pred.shape) > 1:
                    pred = pred[0]
    
    return np.mean(times), pred

def run_hybrid_uint8(data):
    cpu_interp = tf.lite.Interpreter(CPU_UINT8)
    cpu_interp.allocate_tensors()
    cpu_inp = cpu_interp.get_input_details()
    cpu_out = cpu_interp.get_output_details()
    
    times = []
    pred = None
    
    with VDevice() as vdev:
        hef = HEF(NPU_HEF)
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = vdev.configure(hef, cfg)[0]
        
        with ng.activate():
            inp_params = InputVStreamParams.make_from_network_group(ng, quantized=True, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make_from_network_group(ng, quantized=True, format_type=FormatType.UINT8)
            
            with InferVStreams(ng, inp_params, out_params) as infer:
                hef_info = HEF(NPU_HEF)
                inp_name = hef_info.get_input_vstream_infos()[0].name
                out_name = hef_info.get_output_vstream_infos()[0].name
                
                for _ in range(WARMUP):
                    npu_result = infer.infer({inp_name: data})
                    npu_out_uint8 = npu_result[out_name]
                    cpu_interp.set_tensor(cpu_inp[0]['index'], npu_out_uint8)
                    cpu_interp.invoke()
                
                for _ in range(RUNS):
                    start = time.perf_counter()
                    npu_result = infer.infer({inp_name: data})
                    npu_out_uint8 = npu_result[out_name]
                    cpu_interp.set_tensor(cpu_inp[0]['index'], npu_out_uint8)
                    cpu_interp.invoke()
                    times.append((time.perf_counter() - start) * 1000)
                
                pred_uint8 = cpu_interp.get_tensor(cpu_out[0]['index'])[0]
                pred = pred_uint8.astype(np.float32) / 255.0
    
    return np.mean(times), pred

# ==================== Main ====================
print("="*100)
print("최종 종합 벤치마크 + 에너지 측정")
print("="*100)

results = {
    'CPU': {'top1': 0, 'top5': 0, 'total': 0, 'times': [], 'temp_start': None, 'temp_end': None},
    'NPU': {'top1': 0, 'top5': 0, 'total': 0, 'times': [], 'temp_start': None, 'temp_end': None},
    'Hybrid': {'top1': 0, 'top5': 0, 'total': 0, 'times': [], 'temp_start': None, 'temp_end': None}
}

for method_name, run_func in [('CPU', run_cpu_baseline), ('NPU', run_npu_full), ('Hybrid', run_hybrid_uint8)]:
    print(f"\n{'='*100}")
    print(f"{method_name} 벤치마크")
    print(f"{'='*100}")
    
    # 시작 온도 측정
    temp_start = get_cpu_temp()
    freq_start = get_cpu_freq()
    results[method_name]['temp_start'] = temp_start
    
    if temp_start:
        print(f"시작 온도: {temp_start:.1f}°C, 주파수: {freq_start:.0f} MHz")
    
    # 벤치마크 실행
    for idx, img in enumerate(TEST_IMAGES, 1):
        print(f"[{idx}/{len(TEST_IMAGES)}] {img['name']:<20}", end=" ")
        
        data = preprocess(img['path'])
        t, pred = run_func(data)
        
        top1 = np.argmax(pred)
        top5 = np.argsort(pred)[-5:][::-1]
        
        results[method_name]['total'] += 1
        results[method_name]['times'].append(t)
        if top1 == img['label']:
            results[method_name]['top1'] += 1
        if img['label'] in top5:
            results[method_name]['top5'] += 1
        
        print(f"✓ {t:.1f}ms")
    
    # 종료 온도 측정
    temp_end = get_cpu_temp()
    freq_end = get_cpu_freq()
    results[method_name]['temp_end'] = temp_end
    
    if temp_end and temp_start:
        temp_rise = temp_end - temp_start
        print(f"종료 온도: {temp_end:.1f}°C, 주파수: {freq_end:.0f} MHz")
        print(f"온도 상승: {temp_rise:+.1f}°C")

# ==================== 결과 출력 ====================
print(f"\n{'='*100}")
print("전체 평균 - 성능 (Latency)")
print(f"{'='*100}")

print(f"{'Method':<15} {'Time (ms)':<15} {'Speedup'}")
print("-"*100)

cpu_avg = np.mean(results['CPU']['times'])
npu_avg = np.mean(results['NPU']['times'])
hyb_avg = np.mean(results['Hybrid']['times'])

print(f"{'CPU':<15} {cpu_avg:>7.2f} ms{'':<6} {'1.00x'}")
print(f"{'NPU':<15} {npu_avg:>7.2f} ms{'':<6} {cpu_avg/npu_avg:>4.2f}x")
print(f"{'Hybrid':<15} {hyb_avg:>7.2f} ms{'':<6} {cpu_avg/hyb_avg:>4.2f}x")

print(f"\n{'='*100}")
print("전체 평균 - 정확도 (Accuracy)")
print(f"{'='*100}")

print(f"{'Method':<20} {'Top-1 Accuracy':<30} {'Top-5 Accuracy':<30} {'Samples'}")
print("-"*100)

for method in ['CPU', 'NPU', 'Hybrid']:
    data = results[method]
    top1_pct = (data['top1'] / data['total']) * 100
    top5_pct = (data['top5'] / data['total']) * 100
    top1_str = f"{top1_pct:>5.1f}% ({data['top1']}/{data['total']})"
    top5_str = f"{top5_pct:>5.1f}% ({data['top5']}/{data['total']})"
    print(f"{method:<20} {top1_str:<30} {top5_str:<30} {data['total']}")

# ==================== 에너지 분석 ====================
print(f"\n{'='*100}")
print("에너지 효율 분석")
print(f"{'='*100}")

print(f"{'Method':<15} {'온도 상승':<15} {'추정 전력':<15} {'추정 에너지':<20} {'에너지 효율'}")
print("-"*100)

# 전력 추정 (W)
cpu_power = 6.0  # CPU 추론 시 약 6W
npu_power = 2.5  # Hailo-8L 약 2.5W
hybrid_power = 4.0  # NPU + CPU 일부

for method, power in [('CPU', cpu_power), ('NPU', npu_power), ('Hybrid', hybrid_power)]:
    data = results[method]
    avg_time_s = np.mean(data['times']) / 1000  # ms → s
    energy_j = power * avg_time_s  # Joule
    
    temp_start = data.get('temp_start')
    temp_end = data.get('temp_end')
    temp_rise = f"{temp_end - temp_start:+.1f}°C" if (temp_start and temp_end) else "N/A"
    
    efficiency = (cpu_power * cpu_avg / 1000) / energy_j if energy_j > 0 else 0
    
    print(f"{method:<15} {temp_rise:<15} {power:>5.1f} W{'':<8} {energy_j*1000:>8.3f} mJ{'':<10} {efficiency:>4.2f}x")

print(f"\n참고:")
print(f"  - 에너지 = 전력 × 시간")
print(f"  - CPU 전력: ~6W (추정), NPU 전력: ~2.5W (Hailo-8L 스펙)")
print(f"  - 온도 상승은 발열량과 비례")

print(f"\n{'='*100}")
