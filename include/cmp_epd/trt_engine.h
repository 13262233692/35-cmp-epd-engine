#pragma once

#include "common.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace cmp_epd {

struct LstmState {
    std::vector<float> hidden;
    std::vector<float> cell;
    int32_t layers;
    int32_t directions;
    int32_t batch;
    int32_t hidden_size;

    LstmState() : layers(0), directions(0), batch(0), hidden_size(0) {}

    void init(int32_t l, int32_t d, int32_t b, int32_t h) {
        layers = l;
        directions = d;
        batch = b;
        hidden_size = h;
        size_t sz = static_cast<size_t>(l) * d * b * h;
        hidden.assign(sz, 0.0f);
        cell.assign(sz, 0.0f);
    }

    void reset() {
        std::memset(hidden.data(), 0, hidden.size() * sizeof(float));
        std::memset(cell.data(), 0, cell.size() * sizeof(float));
    }

    size_t total_bytes() const {
        return (hidden.size() + cell.size()) * sizeof(float);
    }
};

struct FixedMemoryPool {
    static constexpr int32_t K_SNAPSHOT_POOL_SIZE = 256;

    int32_t snapshot_index;
    int32_t snapshot_count;
    std::vector<std::vector<float>> snapshot_pool;
    size_t snapshot_capacity_bytes;

    FixedMemoryPool()
        : snapshot_index(0)
        , snapshot_count(0)
        , snapshot_capacity_bytes(0) {}

    void init(size_t result_vector_size) {
        snapshot_capacity_bytes = result_vector_size * sizeof(float);
        snapshot_pool.clear();
        snapshot_pool.resize(K_SNAPSHOT_POOL_SIZE);
        for (auto& slot : snapshot_pool) {
            slot.resize(result_vector_size, 0.0f);
        }
        snapshot_index = 0;
        snapshot_count = 0;
    }

    float* acquire_snapshot() {
        float* slot = snapshot_pool[snapshot_index].data();
        snapshot_index = (snapshot_index + 1) % K_SNAPSHOT_POOL_SIZE;
        if (snapshot_count < K_SNAPSHOT_POOL_SIZE) {
            snapshot_count++;
        }
        return slot;
    }

    void reset() {
        snapshot_index = 0;
        snapshot_count = 0;
        for (auto& slot : snapshot_pool) {
            std::memset(slot.data(), 0, slot.size() * sizeof(float));
        }
    }

    size_t total_bytes() const {
        return snapshot_pool.size() * snapshot_capacity_bytes;
    }
};

class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();

    bool init(const EpdConfig& config);

    bool infer(const Tensor3D& input, EpdResult& result);

    bool is_loaded() const { return engine_loaded_; }

    bool reset();

    bool reset_lstm_states();

    bool is_lstm_state_dirty() const { return lstm_state_dirty_; }

    bool reset_output_pool();

    int32_t input_time_steps() const { return input_time_steps_; }
    int32_t input_channels() const { return input_channels_; }
    int32_t input_waves() const { return input_waves_; }

    uint64_t last_inference_us() const { return last_inference_us_; }
    float average_inference_ms() const { return avg_inference_ms_; }

    int32_t lstm_layers() const { return lstm_state_.layers; }
    int32_t lstm_hidden_size() const { return lstm_state_.hidden_size; }

    size_t get_total_allocated_bytes() const;

private:
    bool load_engine(const std::string& engine_path);
    bool allocate_static_buffers();
    bool free_static_buffers();
    bool setup_bindings();
    bool deep_copy_output(EpdResult& result);
    bool deep_copy_raw_output(const void* src_output, size_t output_bytes);

    void* trt_runtime_;
    void* trt_engine_;
    void* trt_context_;

    std::vector<void*> device_buffers_;
    std::vector<void*> host_input_buffers_;
    std::vector<void*> host_output_buffers_;
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
    bool buffers_allocated_;

    LstmState lstm_state_;
    std::atomic<bool> lstm_state_dirty_;

    FixedMemoryPool output_pool_;
    std::atomic<int32_t> pool_ref_count_;

    std::vector<float> last_raw_output_;

    uint64_t last_inference_us_;
    float avg_inference_ms_;
    int32_t inference_count_;
    double total_inference_us_;

    mutable std::mutex inference_mutex_;
};

}
