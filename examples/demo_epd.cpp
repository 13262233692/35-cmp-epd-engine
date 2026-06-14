#include "cmp_epd/epd_engine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace cmp_epd;

SpectrumFrame generate_simulated_frame(int32_t wave_count, double timestamp, float endpoint_progress) {
    SpectrumFrame frame;
    frame.wave_count = wave_count;
    frame.head_index = 0;
    frame.timestamp = timestamp;
    frame.intensities.resize(wave_count);

    float base_intensity = 8000.0f;
    float oscillation = 2000.0f * std::sin(endpoint_progress * 3.14159f * 4.0f);
    float trend = 3000.0f * endpoint_progress;

    for (int32_t i = 0; i < wave_count; ++i) {
        float wave_pos = static_cast<float>(i) / static_cast<float>(wave_count);
        float peak = std::exp(-std::pow(wave_pos - 0.3f - endpoint_progress * 0.4f, 2) * 30.0f);
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 200.0f;
        float slurry_noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 500.0f
            * (1.0f + 0.5f * std::sin(timestamp * 10.0f + i * 0.1f));

        frame.intensities[i] = base_intensity + oscillation * wave_pos
            + peak * 4000.0f + trend * wave_pos + noise + slurry_noise;
    }

    return frame;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CMP EPD Engine Demo" << std::endl;
    std::cout << "  化学机械抛光 在线终点检测引擎" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    EpdConfig config;
    config.wave_channels = 512;
    config.time_window = 32;
    config.polish_heads = 1;
    config.sample_rate_hz = 20;
    config.sg_window_size = 11;
    config.sg_poly_order = 3;
    config.endpoint_threshold = 0.85f;
    config.enable_cuda = false;
    config.gpu_device_id = 0;
    config.trt_engine_path = "";

    std::cout << "[配置] 波段数: " << config.wave_channels << std::endl;
    std::cout << "[配置] 时间窗口: " << config.time_window << " 帧" << std::endl;
    std::cout << "[配置] 采样率: " << config.sample_rate_hz << " Hz" << std::endl;
    std::cout << "[配置] S-G 滤波窗口: " << config.sg_window_size << std::endl;
    std::cout << "[配置] S-G 多项式阶数: " << config.sg_poly_order << std::endl;
    std::cout << "[配置] 终点阈值: " << config.endpoint_threshold << std::endl;
    std::cout << std::endl;

    EpdEngine engine;
    if (!engine.init(config)) {
        std::cerr << "错误: EPD 引擎初始化失败" << std::endl;
        return -1;
    }
    std::cout << "[初始化] EPD 引擎初始化成功" << std::endl;
    std::cout << std::endl;

    int32_t total_frames = 200;
    int32_t endpoint_start_frame = 120;
    float sample_interval = 1.0f / config.sample_rate_hz;

    std::cout << "[模拟] 开始模拟抛光过程..." << std::endl;
    std::cout << "[模拟] 总帧数: " << total_frames << std::endl;
    std::cout << "[模拟] 终点起始帧: " << endpoint_start_frame << std::endl;
    std::cout << std::endl;

    int32_t detected_endpoint_frame = -1;
    float min_prob = 1.0f;
    float max_prob = 0.0f;

    for (int32_t i = 0; i < total_frames; ++i) {
        double timestamp = static_cast<double>(i) * sample_interval;

        float endpoint_progress = 0.0f;
        if (i < endpoint_start_frame) {
            endpoint_progress = static_cast<float>(i) / static_cast<float>(endpoint_start_frame) * 0.3f;
        } else {
            endpoint_progress = 0.3f + (static_cast<float>(i - endpoint_start_frame)
                / static_cast<float>(total_frames - endpoint_start_frame)) * 0.7f;
        }

        SpectrumFrame frame = generate_simulated_frame(
            config.wave_channels, timestamp, endpoint_progress
        );

        EpdResult result;
        EpdStatus status = engine.process_frame(frame, result);

        if (result.endpoint_probability < min_prob) min_prob = result.endpoint_probability;
        if (result.endpoint_probability > max_prob) max_prob = result.endpoint_probability;

        if (i % 20 == 0 || i == total_frames - 1) {
            std::cout << "[帧 " << std::setw(4) << i << "] ";
            std::cout << "时间: " << std::fixed << std::setprecision(2) << timestamp << "s | ";
            std::cout << "终点概率: " << std::fixed << std::setprecision(4)
                << result.endpoint_probability << " | ";
            std::cout << "厚度: " << std::fixed << std::setprecision(1)
                << result.thickness_estimate_nm << " nm | ";
            std::cout << "状态: ";

            switch (status) {
                case EpdStatus::OK:
                    std::cout << "就绪";
                    break;
                case EpdStatus::NOT_READY:
                    std::cout << "缓冲中";
                    break;
                case EpdStatus::ERROR_INFERENCE:
                    std::cout << "推理错误";
                    break;
                default:
                    std::cout << "未知";
            }

            if (result.is_endpoint && detected_endpoint_frame < 0) {
                detected_endpoint_frame = i;
                std::cout << " <-- 终点检测到!";
            }

            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  统计信息" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "处理总帧数: " << engine.total_frames_processed() << std::endl;
    std::cout << "平均延迟: " << std::fixed << std::setprecision(3)
        << engine.avg_latency_ms() << " ms/帧" << std::endl;
    std::cout << "推理延迟: " << std::fixed << std::setprecision(3)
        << engine.inference_latency_ms() << " ms/次" << std::endl;
    std::cout << "最小概率: " << std::fixed << std::setprecision(4) << min_prob << std::endl;
    std::cout << "最大概率: " << std::fixed << std::setprecision(4) << max_prob << std::endl;

    if (detected_endpoint_frame >= 0) {
        std::cout << "终点检测帧: " << detected_endpoint_frame << std::endl;
        std::cout << "终点检测时间: " << std::fixed << std::setprecision(2)
            << static_cast<double>(detected_endpoint_frame) * sample_interval << " s" << std::endl;
    } else {
        std::cout << "未检测到终点" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "演示完成。" << std::endl;

    return 0;
}
