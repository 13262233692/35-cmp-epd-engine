#include "cmp_epd/epd_engine.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>

namespace cmp_epd {

EpdEngine::EpdEngine()
    : initialized_(false)
    , ready_(false)
    , total_frames_(0)
    , avg_latency_ms_(0.0f)
    , total_latency_us_(0.0)
    , prob_smooth_window_(5)
    , endpoint_streak_(0)
    , streak_threshold_(3)
    , endpoint_threshold_(0.85f)
    , batch_start_frame_(0)
    , lstm_reset_required_(false) {
}

EpdEngine::~EpdEngine() {
    reset();
}

bool EpdEngine::init(const EpdConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    config_ = config;
    endpoint_threshold_ = config.endpoint_threshold;

    reorganizer_ = std::make_unique<TensorReorganizer>();
    if (!reorganizer_->init(config)) {
        std::cerr << "[EPD] Failed to initialize tensor reorganizer" << std::endl;
        return false;
    }

    trt_engine_ = std::make_unique<TensorRTEngine>();
    if (!trt_engine_->init(config)) {
        std::cerr << "[EPD] Failed to initialize TensorRT engine" << std::endl;
        return false;
    }

    initialized_ = true;
    ready_ = false;

    reset_internal_state(true);

    return true;
}

bool EpdEngine::load_engine(const std::string& engine_path) {
    if (!initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    EpdConfig new_config = config_;
    new_config.trt_engine_path = engine_path;

    trt_engine_ = std::make_unique<TensorRTEngine>();
    if (!trt_engine_->init(new_config)) {
        return false;
    }

    config_.trt_engine_path = engine_path;

    lstm_reset_required_ = true;

    return true;
}

bool EpdEngine::is_ready() const {
    if (!initialized_) return false;
    return ready_;
}

void EpdEngine::set_endpoint_threshold(float threshold) {
    endpoint_threshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

bool EpdEngine::begin_batch(const std::string& batch_id,
                           const std::string& recipe_id,
                           int32_t wafer_index,
                           int32_t zone_index) {
    if (!initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    end_batch();

    current_batch_.batch_id = batch_id;
    current_batch_.recipe_id = recipe_id;
    current_batch_.wafer_index = wafer_index;
    current_batch_.zone_index = zone_index;
    current_batch_.start_timestamp = static_cast<double>(current_time_us()) / 1e6;
    current_batch_.frames_processed = 0;

    reset_internal_state(false);

    if (trt_engine_) {
        trt_engine_->reset_lstm_states();
        trt_engine_->reset_output_pool();
    }
    lstm_reset_required_ = false;

    batch_start_frame_ = total_frames_.load();

    std::cout << "[EPD] Batch started: batch=" << batch_id
              << " wafer=" << wafer_index
              << " zone=" << zone_index << std::endl;

    return true;
}

bool EpdEngine::end_batch() {
    if (current_batch_.is_active()) {
        std::cout << "[EPD] Batch ended: batch=" << current_batch_.batch_id
                  << " frames=" << current_batch_.frames_processed << std::endl;
    }

    current_batch_.reset();

    reset_internal_state(false);

    if (trt_engine_) {
        trt_engine_->reset_lstm_states();
        trt_engine_->reset_output_pool();
    }
    lstm_reset_required_ = true;

    return true;
}

bool EpdEngine::switch_zone(int32_t new_zone_index) {
    if (!initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!current_batch_.is_active()) {
        return false;
    }

    current_batch_.zone_index = new_zone_index;

    reorganizer_->reset();
    endpoint_streak_ = 0;
    prob_history_.clear();

    if (trt_engine_) {
        trt_engine_->reset_lstm_states();
    }

    std::cout << "[EPD] Zone switched: wafer=" << current_batch_.wafer_index
              << " new_zone=" << new_zone_index << std::endl;

    return true;
}

bool EpdEngine::switch_wafer(int32_t new_wafer_index) {
    if (!initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!current_batch_.is_active()) {
        return false;
    }

    current_batch_.wafer_index = new_wafer_index;
    current_batch_.frames_processed = 0;

    reset_internal_state(false);

    if (trt_engine_) {
        trt_engine_->reset_lstm_states();
        trt_engine_->reset_output_pool();
    }

    batch_start_frame_ = total_frames_.load();

    std::cout << "[EPD] Wafer switched: batch=" << current_batch_.batch_id
              << " new_wafer=" << new_wafer_index << std::endl;

    return true;
}

EpdStatus EpdEngine::process_frame(const SpectrumFrame& frame, EpdResult& result) {
    if (!initialized_) {
        return EpdStatus::ERROR_INIT;
    }

    if (static_cast<int32_t>(frame.intensities.size()) != config_.wave_channels) {
        return EpdStatus::ERROR_INVALID_INPUT;
    }

    uint64_t start_time = current_time_us();

    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (lstm_reset_required_ && trt_engine_) {
            trt_engine_->reset_lstm_states();
            trt_engine_->reset_output_pool();
            lstm_reset_required_ = false;
        }

        if (!reorganizer_->push_frame(frame)) {
            return EpdStatus::ERROR_INVALID_INPUT;
        }

        if (!reorganizer_->is_full()) {
            result = last_result_;
            result.timestamp = static_cast<double>(current_time_us()) / 1e6;
            ready_ = false;

            if (current_batch_.is_active()) {
                current_batch_.frames_processed++;
            }
            total_frames_++;
            return EpdStatus::NOT_READY;
        }

        Tensor3D tensor;
        if (!reorganizer_->get_tensor(tensor)) {
            return EpdStatus::ERROR_INFERENCE;
        }

        EpdResult infer_result;
        if (!trt_engine_->infer(tensor, infer_result)) {
            return EpdStatus::ERROR_INFERENCE;
        }

        smooth_probability(infer_result.endpoint_probability);

        infer_result.is_endpoint = check_endpoint(infer_result) == EpdStatus::OK;

        if (infer_result.is_endpoint) {
            endpoint_streak_++;
        } else {
            endpoint_streak_ = 0;
        }

        infer_result.is_endpoint = (endpoint_streak_ >= streak_threshold_);

        last_result_ = infer_result;
        result = infer_result;

        snapshot_buffer_.push(infer_result);

        if (current_batch_.is_active()) {
            current_batch_.frames_processed++;
        }
    }

    total_frames_++;
    update_statistics(current_time_us() - start_time);

    ready_ = true;
    return EpdStatus::OK;
}

bool EpdEngine::smooth_probability(float& prob) {
    prob_history_.push_back(prob);
    if (static_cast<int32_t>(prob_history_.size()) > prob_smooth_window_) {
        prob_history_.pop_front();
    }

    if (prob_history_.empty()) {
        return false;
    }

    float sum = 0.0f;
    for (float p : prob_history_) {
        sum += p;
    }
    prob = sum / static_cast<float>(prob_history_.size());

    return true;
}

EpdStatus EpdEngine::check_endpoint(const EpdResult& result) const {
    if (result.endpoint_probability >= endpoint_threshold_ &&
        result.confidence > 0.5f) {
        return EpdStatus::OK;
    }
    return EpdStatus::NOT_READY;
}

EpdStatus EpdEngine::get_last_result(EpdResult& result) const {
    if (!initialized_) {
        return EpdStatus::ERROR_INIT;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    result = last_result_;
    return ready_ ? EpdStatus::OK : EpdStatus::NOT_READY;
}

bool EpdEngine::get_snapshot_history(std::vector<EpdResult>& history, int32_t max_count) const {
    std::lock_guard<std::mutex> lock(mutex_);

    int32_t n = std::min(max_count, snapshot_buffer_.count);
    if (n <= 0) {
        history.clear();
        return true;
    }

    history.resize(static_cast<size_t>(n));

    int32_t total = snapshot_buffer_.count;
    int32_t start_idx;

    if (total < SnapshotRingBuffer::K_MAX_SNAPSHOTS) {
        start_idx = 0;
    } else {
        start_idx = snapshot_buffer_.write_idx;
    }

    int32_t skip = std::max(0, total - n);
    for (int32_t i = 0; i < n; ++i) {
        int32_t buf_idx = (start_idx + skip + i) % SnapshotRingBuffer::K_MAX_SNAPSHOTS;
        const auto& snap = snapshot_buffer_.buffer[buf_idx];
        history[i].endpoint_probability = snap.endpoint_prob;
        history[i].thickness_estimate_nm = snap.thickness_nm;
        history[i].timestamp = snap.timestamp;
        history[i].is_endpoint = snap.is_endpoint;
        history[i].confidence = snap.endpoint_prob;
    }

    return true;
}

size_t EpdEngine::get_total_memory_allocated() const {
    size_t total = 0;

    total += snapshot_buffer_.total_bytes();
    if (trt_engine_) {
        total += trt_engine_->get_total_allocated_bytes();
    }
    total += prob_history_.size() * sizeof(float);

    return total;
}

void EpdEngine::update_statistics(uint64_t latency_us) {
    total_latency_us_ += static_cast<double>(latency_us);
    uint64_t frames = total_frames_.load();
    if (frames > 0) {
        avg_latency_ms_ = static_cast<float>(
            total_latency_us_ / static_cast<double>(frames) / 1000.0
        );
    }
}

bool EpdEngine::reset_internal_state(bool full_reset) {
    if (reorganizer_) {
        reorganizer_->reset();
    }

    ready_ = false;
    endpoint_streak_ = 0;
    prob_history_.clear();

    last_result_.endpoint_probability = 0.0f;
    last_result_.thickness_estimate_nm = 0.0f;
    last_result_.timestamp = 0.0;
    last_result_.is_endpoint = false;
    last_result_.confidence = 0.0f;

    snapshot_buffer_.reset();

    if (full_reset) {
        total_frames_ = 0;
        avg_latency_ms_ = 0.0f;
        total_latency_us_ = 0.0;
        batch_start_frame_ = 0;
    }

    lstm_reset_required_ = true;

    return true;
}

void EpdEngine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    current_batch_.reset();

    if (reorganizer_) {
        reorganizer_->reset();
    }
    if (trt_engine_) {
        trt_engine_->reset();
    }

    reset_internal_state(true);

    initialized_ = false;
}

}
