#pragma once

#include "common.h"
#include "sg_filter.h"
#include "tensor_reorganizer.h"
#include "trt_engine.h"
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <deque>

namespace cmp_epd {

class EpdEngine {
public:
    EpdEngine();
    ~EpdEngine();

    bool init(const EpdConfig& config);

    EpdStatus process_frame(const SpectrumFrame& frame, EpdResult& result);

    EpdStatus get_last_result(EpdResult& result) const;

    bool is_ready() const;
    bool is_initialized() const { return initialized_; }

    void set_endpoint_threshold(float threshold);
    float endpoint_threshold() const { return endpoint_threshold_; }

    void reset();

    int32_t wave_channels() const { return config_.wave_channels; }
    int32_t time_window() const { return config_.time_window; }
    int32_t polish_heads() const { return config_.polish_heads; }

    uint64_t total_frames_processed() const { return total_frames_; }
    float avg_latency_ms() const { return avg_latency_ms_; }
    float inference_latency_ms() const { return trt_engine_ ? trt_engine_->average_inference_ms() : 0.0f; }

    const EpdConfig& config() const { return config_; }

    bool load_engine(const std::string& engine_path);

    EpdStatus check_endpoint(const EpdResult& result) const;

private:
    void update_statistics(uint64_t latency_us);
    bool smooth_probability(float& prob);

    EpdConfig config_;
    bool initialized_;
    bool ready_;

    std::unique_ptr<TensorReorganizer> reorganizer_;
    std::unique_ptr<TensorRTEngine> trt_engine_;

    EpdResult last_result_;
    std::atomic<uint64_t> total_frames_;
    float avg_latency_ms_;
    double total_latency_us_;

    std::deque<float> prob_history_;
    int32_t prob_smooth_window_;

    mutable std::mutex mutex_;

    int32_t endpoint_streak_;
    int32_t streak_threshold_;
    float endpoint_threshold_;
};

}
