# Block Partitioning-based CPU-NPU Hybrid Inference System

Raspberry Pi 5 + Hailo-8L NPU를 활용한 하이브리드 AI 추론 시스템

## 🎯 실험 목표

MobileNetV1 이미지 분류를 CPU, NPU, Hybrid 방식으로 실행하여 성능, 정확도, 에너지 효율을 비교 분석

## 📊 주요 결과

| 방식 | Latency | Speedup | Top-1 정확도 |
|------|---------|---------|--------------|
| **CPU** | 47.61ms | 1.00x | 29.5% |
| **NPU** | 2.06ms | **23.21x** | 27.3% |
| **Hybrid** | 7.51ms | 6.34x | 0%* |


## 🛠️ 시스템 구성

- **Hardware**: Raspberry Pi 5 (8GB)
- **NPU**: Hailo-8L (13 TOPS)
- **Model**: MobileNetV1 (ImageNet pretrained)
- **Framework**: TensorFlow 2.15, Hailo Dataflow Compiler

## 📁 파일 구조
```
├── model_preparation.py    # 모델 생성 및 양자화 스크립트 (WSL)
├── benchmark.py            # 성능/에너지 벤치마크 스크립트 (Raspberry Pi)
├── results.txt             # 실험 결과 상세
└── test_images/            # 테스트 샘플 이미지
```

## 🚀 실행 방법

### 1. 모델 준비 (WSL/Ubuntu)
```bash
python3 model_preparation.py
```
출력: `cpu_blocks_9_end_uint8.tflite`

### 2. 벤치마크 실행 (Raspberry Pi + Hailo)
```bash
python3 benchmark.py
```

## 📖 참고

- [Hailo-8L Datasheet](https://hailo.ai/products/hailo-8l/)
- [MobileNets Paper](https://arxiv.org/abs/1704.04861)
- Raspberry Pi 5: ARM Cortex-A76

---

**졸업 프로젝트 | 2025년 2학기**
