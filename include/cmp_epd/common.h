#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

namespace cmp_epd {

constexpr int32_t DEFAULT_WAVE_CHANNELS = 512;
constexpr int32_t DEFAULT_SAMPLE_RATE_HZ = 20;
constexpr int32_t DEFAULT_TIME_WINDOW = 32;
constexpr int32_t DEFAULT_POLISH_HEADS = 1;
constexpr int32_t BITS_PER_SAMPLE = 14;
constexpr float MAX_ADC_VALUE = 16383.0f;

struct SpectrumFrame {
    std::vector<float> intensities;
    double timestamp;
    int32_t head_index;
    int32_t wave_count;
};

struct Tensor3D {
    int32_t time_steps;
    int32_t channels;
    int32_t waves;
    std::vector<float> data;

    Tensor3D() : time_steps(0), channels(0), waves(0) {}

    Tensor3D(int32_t t, int32_t c, int32_t w)
        : time_steps(t), channels(c), waves(w), data(t * c * w, 0.0f) {}

    float& at(int32_t t, int32_t c, int32_t w) {
        return data[t * channels * waves + c * waves + w];
    }

    const float& at(int32_t t, int32_t c, int32_t w) const {
        return data[t * channels * waves + c * waves + w];
    }

    size_t size() const { return data.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct EpdResult {
    float endpoint_probability;
    float thickness_estimate_nm;
    double timestamp;
    bool is_endpoint;
    float confidence;
};

enum class EpdStatus {
    OK = 0,
    ERROR_INIT = 1,
    ERROR_INFERENCE = 2,
    ERROR_INVALID_INPUT = 3,
    ERROR_MODEL_LOAD = 4,
    NOT_READY = 5
};

struct EpdConfig {
    int32_t wave_channels = DEFAULT_WAVE_CHANNELS;
    int32_t sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ;
    int32_t time_window = DEFAULT_TIME_WINDOW;
    int32_t polish_heads = DEFAULT_POLISH_HEADS;
    int32_t sg_window_size = 11;
    int32_t sg_poly_order = 3;
    float endpoint_threshold = 0.85f;
    std::string trt_engine_path;
    bool enable_cuda = true;
    int32_t gpu_device_id = 0;
};

inline uint64_t current_time_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count()
    );
}

}
