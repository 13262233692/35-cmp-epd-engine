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
#include <vector>
#include <cstdint>

namespace cmp_epd {

struct BatchInfo {
    std::string batch_id;
    std::string recipe_id;
    int32_t wafer_index;
    int32_t zone_index;
    double start_timestamp;
    int64_t frames_processed;

    BatchInfo()
        : batch_id("")
        , recipe_id("")
        , wafer_index(-1)
        , zone_index(-1)
        , start_timestamp(0.0)
        , frames_processed(0) {}

    void reset() {
        batch_id.clear();
        recipe_id.clear();
        wafer_index = -1;
        zone_index = -1;
        start_timestamp = 0.0;
        frames_processed = 0;
    }

    bool is_active() const {
        return wafer_index >= 0;
    }
};

struct SnapshotRingBuffer {
    static constexpr int32_t K_MAX_SNAPSHOTS = 1024;

    struct Snapshot {
        float endpoint_prob;
        float thickness_nm;
        double timestamp;
        bool is_endpoint;
        uint64_t frame_seq;
    };

    std::vector<Snapshot> buffer;
    int32_t write_idx;
    int32_t count;
    std::atomic<uint64_t> seq_counter;

    SnapshotRingBuffer() : write_idx(0), count(0), seq_counter(0) {
        buffer.resize(K_MAX_SNAPSHOTS);
    }

    void push(const EpdResult& result) {
        Snapshot& slot = buffer[write_idx];
        slot.endpoint_prob = result.endpoint_probability;
        slot.thickness_nm = result.thickness_estimate_nm;
        slot.timestamp = result.timestamp;
        slot.is_endpoint = result.is_endpoint;
        slot.frame_seq = seq_counter.fetch_add(1);

        write_idx = (write_idx + 1) % K_MAX_SNAPSHOTS;
        if (count < K_MAX_SNAPSHOTS) {
            count++;
        }
    }

    void reset() {
        write_idx = 0;
        count = 0;
        seq_counter = 0;
        for (auto& s : buffer) {
            s.endpoint_prob = 0.0f;
            s.thickness_nm = 0.0f;
            s.timestamp = 0.0;
            s.is_endpoint = false;
            s.frame_seq = 0;
        }
    }

    size_t total_bytes() const {
        return buffer.size() * sizeof(Snapshot);
    }
};

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

    bool begin_batch(const std::string& batch_id,
                     const std::string& recipe_id,
                     int32_t wafer_index,
                     int32_t zone_index);

    bool end_batch();

    bool switch_zone(int32_t new_zone_index);

    bool switch_wafer(int32_t new_wafer_index);

    const BatchInfo& current_batch() const { return current_batch_; }
    bool is_batch_active() const { return current_batch_.is_active(); }

    int32_t get_snapshot_count() const { return snapshot_buffer_.count; }

    bool get_snapshot_history(std::vector<EpdResult>& history, int32_t max_count = 256) const;

    int32_t wave_channels() const { return config_.wave_channels; }
    int32_t time_window() const { return config_.time_window; }
    int32_t polish_heads() const { return config_.polish_heads; }

    uint64_t total_frames_processed() const { return total_frames_; }
    float avg_latency_ms() const { return avg_latency_ms_; }
    float inference_latency_ms() const { return trt_engine_ ? trt_engine_->average_inference_ms() : 0.0f; }

    size_t get_total_memory_allocated() const;

    const EpdConfig& config() const { return config_; }

    bool load_engine(const std::string& engine_path);

    EpdStatus check_endpoint(const EpdResult& result) const;

private:
    void update_statistics(uint64_t latency_us);
    bool smooth_probability(float& prob);
    bool reset_internal_state(bool full_reset = false);

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

    BatchInfo current_batch_;

    SnapshotRingBuffer snapshot_buffer_;

    uint64_t batch_start_frame_;
    bool lstm_reset_required_;
};

}
