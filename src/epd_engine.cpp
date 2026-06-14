#include "cmp_epd/epd_engine.h"
#include <algorithm>
#include <cmath>
#include <iostream>

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
    , endpoint_threshold_(0.85f) {
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
    total_frames_ = 0;
    endpoint_streak_ = 0;
    prob_history_.clear();

    last_result_.endpoint_probability = 0.0f;
    last_result_.thickness_estimate_nm = 0.0f;
    last_result_.timestamp = 0.0;
    last_result_.is_endpoint = false;
    last_result_.confidence = 0.0f;

    return true;
}

bool EpdEngine::load_engine(const std::string& engine_path) {
    if (!initialized_) {
        return false;
    }

    EpdConfig new_config = config_;
    new_config.trt_engine_path = engine_path;

    trt_engine_ = std::make_unique<TensorRTEngine>();
    if (!trt_engine_->init(new_config)) {
        return false;
    }

    config_.trt_engine_path = engine_path;
    return true;
}

bool EpdEngine::is_ready() const {
    if (!initialized_) return false;
    return ready_;
}

void EpdEngine::set_endpoint_threshold(float threshold) {
    endpoint_threshold_ = std::max(0.0f, std::min(1.0f, threshold));
}

EpdStatus EpdEngine::process_frame(const SpectrumFrame& frame, EpdResult& result) {
    if (!initialized_) {
        return EpdStatus::ERROR_INIT;
    }

    if (static_cast<int32_t>(frame.intensities.size()) != config_.wave_channels) {
        return EpdStatus::ERROR_INVALID_INPUT;
    }

    uint64_t start_time = current_time_us();

    if (!reorganizer_->push_frame(frame)) {
        return EpdStatus::ERROR_INVALID_INPUT;
    }

    if (!reorganizer_->is_full()) {
        result = last_result_;
        result.timestamp = static_cast<double>(current_time_us()) / 1e6;
        ready_ = false;
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

    if (endpoint_streak_ >= streak_threshold_) {
        infer_result.is_endpoint = true;
    } else {
        infer_result.is_endpoint = false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_result_ = infer_result;
    }

    result = infer_result;

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

void EpdEngine::update_statistics(uint64_t latency_us) {
    total_latency_us_ += static_cast<double>(latency_us);
    avg_latency_ms_ = static_cast<float>(
        total_latency_us_ / static_cast<double>(total_frames_.load()) / 1000.0
    );
}

void EpdEngine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (reorganizer_) {
        reorganizer_->reset();
    }
    if (trt_engine_) {
        trt_engine_->reset();
    }

    initialized_ = false;
    ready_ = false;
    total_frames_ = 0;
    avg_latency_ms_ = 0.0f;
    total_latency_us_ = 0.0;
    endpoint_streak_ = 0;
    prob_history_.clear();
}

}
