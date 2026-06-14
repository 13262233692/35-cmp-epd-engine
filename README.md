# CMP EPD Engine (化学机械抛光在线终点检测引擎)

面向半导体前道制程纳米级膜厚控制的全自动化学机械抛光（CMP）在线终点检测（EPD）及终点拦截引擎，运行于机台边缘计算盒子。

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CMP EPD Engine                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  光谱数据采集  │───▶│  前处理级联   │───▶│  推理引擎    │  │
│  │  (20Hz 512ch)│    │  (S-G + 张量)│    │  (TensorRT) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                            │                               │
│                            ▼                               │
│                      ┌──────────┐                           │
│                      │ 终点判断 │                           │
│                      └──────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 核心特性

### 1. 多通道光谱反射率时空张量重组
- **高密光谱流处理**：支持 512 波段 14-bit 光学相干光谱，20Hz 刷新率
- **Savitzky-Golay 多项式滤波**：一维 S-G 平滑滤波，有效滤除抛光液流动毛刺
- **时空张量重组**：按抛光头摆动拓扑在时间维度堆叠，重组为 3D 动态特征张量 (Time, Channels, Waves)
- **实时归一化**：Z-score 标准化，提升模型推理稳定性

### 2. TensorRT 终点检测前向推理
- **CNN-BiLSTM 混合架构**：1D 卷积捕捉局部多光谱物理关联，BiLSTM 捕捉时序多级特征
- **TensorRT 加速**：FP32/FP16 量化推理，微秒级吞吐
- **流式预测**：每一帧独立推理，实时输出磨穿概率

### 3. 终点检测与拦截
- **概率平滑**：滑动窗口平均滤波，抑制误检
- **连续命中判定**：多帧连续高概率触发终点信号，避免单点误判
- **可配置阈值**：终点概率阈值、连续帧数阈值可动态调整

## 项目结构

```
35-cmp-epd-engine/
├── CMakeLists.txt              # CMake 构建配置
├── README.md                   # 项目文档
├── include/
│   └── cmp_epd/
│       ├── common.h            # 公共类型定义
│       ├── sg_filter.h         # Savitzky-Golay 滤波器
│       ├── tensor_reorganizer.h # 张量重组器
│       ├── trt_engine.h        # TensorRT 推理引擎
│       └── epd_engine.h        # EPD 核心引擎
├── src/
│   ├── sg_filter.cpp
│   ├── tensor_reorganizer.cpp
│   ├── trt_engine.cpp
│   └── epd_engine.cpp
├── examples/
│   └── demo_epd.cpp            # 演示程序
├── configs/
│   └── epd_config.json         # 配置文件示例
└── models/                     # 模型存放目录
```

## 核心模块说明

### [common.h](include/cmp_epd/common.h)
核心数据结构定义：
- `SpectrumFrame`: 单帧光谱数据
- `Tensor3D`: 3D 特征张量 (Time, Channels, Waves)
- `EpdResult`: 终点检测结果
- `EpdConfig`: 引擎配置参数

### [sg_filter.h](include/cmp_epd/sg_filter.h) / [sg_filter.cpp](src/sg_filter.cpp)
Savitzky-Golay 一维多项式平滑滤波器：
- 基于 Gram 多项式的 S-G 权重计算
- 边缘填充模式可选
- 可配置窗口大小和多项式阶数

### [tensor_reorganizer.h](include/cmp_epd/tensor_reorganizer.h) / [tensor_reorganizer.cpp](src/tensor_reorganizer.cpp)
光谱数据时空张量重组器：
- 滑动时间窗口管理
- S-G 滤波 + 归一化预处理流水线
- 线程安全的帧缓冲队列
- 3D 张量 (Time, Channels, Waves) 输出

### [trt_engine.h](include/cmp_epd/trt_engine.h) / [trt_engine.cpp](src/trt_engine.cpp)
TensorRT 推理引擎封装：
- TensorRT Engine 序列化文件加载
- CUDA 显存/主机内存管理
- CNN-BiLSTM 模型前向推理
- 推理性能统计

### [epd_engine.h](include/cmp_epd/epd_engine.h) / [epd_engine.cpp](src/epd_engine.cpp)
EPD 终点检测核心引擎：
- 端到端处理流水线
- 概率平滑与连续命中判定
- 性能统计与延迟监控
- 统一对外接口

## 快速开始

### 依赖要求
- C++17 编译器
- CUDA Toolkit 11.x+
- TensorRT 8.x+
- CMake 3.18+

### 编译构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行演示

```bash
./demo_epd
```

## 使用示例

```cpp
#include "cmp_epd/epd_engine.h"

using namespace cmp_epd;

int main() {
    EpdConfig config;
    config.wave_channels = 512;
    config.time_window = 32;
    config.sg_window_size = 11;
    config.sg_poly_order = 3;
    config.endpoint_threshold = 0.85f;
    config.trt_engine_path = "models/epd_cnn_bilstm.engine";

    EpdEngine engine;
    engine.init(config);

    while (true) {
        SpectrumFrame frame = get_spectrum_from_sensor();

        EpdResult result;
        EpdStatus status = engine.process_frame(frame, result);

        if (status == EpdStatus::OK && result.is_endpoint) {
            trigger_endpoint_interlock();
            break;
        }
    }

    return 0;
}
```

## 模型架构 (CNN-BiLSTM)

```
Input: [Batch, Time=32, Channels=1, Waves=512]
   │
   ├─► Conv1d(32→64, kernel=5) + ReLU + MaxPool
   ├─► Conv1d(64→128, kernel=3) + ReLU + MaxPool
   │
   ├─► Reshape → [Batch, Time, Features]
   │
   ├─► BiLSTM(hidden=128, layers=2)
   │
   ├─► Global Average Pooling
   ├─► Linear(256→64) + ReLU + Dropout
   └─► Linear(64→2) → [endpoint_prob, thickness]
```

## 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 单帧延迟 | < 500 μs | 包含预处理+推理 |
| 推理吞吐 | > 2000 FPS | 批量模式下 |
| 终点检测延迟 | < 200 ms | 端到端延迟 |
| 误检率 | < 0.1% | 连续 3 帧判定 |

## API 参考

### EpdEngine

| 方法 | 说明 |
|------|------|
| `init(config)` | 初始化引擎 |
| `process_frame(frame, result)` | 处理单帧光谱数据 |
| `get_last_result(result)` | 获取最近一次结果 |
| `set_endpoint_threshold(th)` | 设置终点阈值 |
| `reset()` | 重置引擎状态 |
| `load_engine(path)` | 加载/切换 TensorRT 模型 |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| wave_channels | 512 | 光谱波段数 |
| sample_rate_hz | 20 | 采样频率 (Hz) |
| time_window | 32 | 时间窗口帧数 |
| polish_heads | 1 | 抛光头数量 |
| sg_window_size | 11 | S-G 滤波窗口大小 |
| sg_poly_order | 3 | S-G 多项式阶数 |
| endpoint_threshold | 0.85 | 终点概率阈值 |
| gpu_device_id | 0 | GPU 设备 ID |

## License

专有软件，仅限授权使用。
