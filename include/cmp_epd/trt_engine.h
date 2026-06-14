#pragma once

#include "common.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace cmp_epd {

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    bool init(const EpdConfig& config);

    bool infer(const Tensor3D& input, EpdResult& result);

    bool is_loaded() const { return engine_loaded_; }

    bool reset();

    int32_t input_time_steps() const { return input_time_steps_; }
    int32_t input_channels() const { return input_channels_; }
    int32_t input_waves() const { return input_waves_; }

    uint64_t last_inference_us() const { return last_inference_us_; }
    float average_inference_ms() const { return avg_inference_ms_; }

private:
    bool load_engine(const std::string& engine_path);
    bool allocate_buffers();
    bool free_buffers();
    bool setup_bindings();

    void* cuda_runtime_;
    void* trt_runtime_;
    void* trt_engine_;
    void* trt_context_;

    std::vector<void*> device_buffers_;
    std::vector<void*> host_buffers_;
    std::vector<size_t> buffer_sizes_;
    std::vector<int32_t> input_indices_;
    std::vector<int32_t> output_indices_;

    int32_t input_time_steps_;
    int32_t input_channels_;
    int32_t input_waves_;
    int32_t output_size_;

    int32_t gpu_device_id_;
    bool engine_loaded_;
    bool use_cuda_;

    uint64_t last_inference_us_;
    float avg_inference_ms_;
    int32_t inference_count_;
    double total_inference_us_;
};

}
