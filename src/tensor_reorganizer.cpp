#include "cmp_epd/tensor_reorganizer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cmp_epd {

TensorReorganizer::TensorReorganizer()
    : time_window_(0)
    , channels_(0)
    , waves_(0)
    , initialized_(false)
    , normalize_(true) {
}

bool TensorReorganizer::init(const EpdConfig& config) {
    time_window_ = config.time_window;
    channels_ = config.polish_heads;
    waves_ = config.wave_channels;

    if (time_window_ <= 0 || channels_ <= 0 || waves_ <= 0) {
        return false;
    }

    sg_filter_ = std::make_unique<SavitzkyGolayFilter>();
    if (!sg_filter_->init(config.sg_window_size, config.sg_poly_order, waves_)) {
        return false;
    }

    frame_buffer_.clear();
    initialized_ = true;
    return true;
}

bool TensorReorganizer::validate_frame(const SpectrumFrame& frame) const {
    if (static_cast<int32_t>(frame.intensities.size()) != waves_) {
        return false;
    }
    if (frame.wave_count != waves_) {
        return false;
    }
    if (frame.head_index < 0 || frame.head_index >= channels_) {
        return false;
    }
    return true;
}

void TensorReorganizer::normalize_spectrum(std::vector<float>& spectrum) const {
    if (!normalize_ || spectrum.empty()) {
        return;
    }

    float sum = 0.0f;
    for (float v : spectrum) {
        sum += v;
    }
    float mean = sum / static_cast<float>(spectrum.size());

    float sq_sum = 0.0f;
    for (float v : spectrum) {
        float diff = v - mean;
        sq_sum += diff * diff;
    }
    float std = std::sqrt(sq_sum / static_cast<float>(spectrum.size()));

    if (std > 1e-6f) {
        for (float& v : spectrum) {
            v = (v - mean) / std;
        }
    }
}

bool TensorReorganizer::push_frame(const SpectrumFrame& frame) {
    if (!initialized_) {
        return false;
    }
    if (!validate_frame(frame)) {
        return false;
    }

    std::vector<float> filtered(waves_);
    if (!sg_filter_->filter(frame.intensities, filtered)) {
        return false;
    }

    normalize_spectrum(filtered);

    std::lock_guard<std::mutex> lock(mutex_);
    frame_buffer_.push_back(std::move(filtered));

    if (static_cast<int32_t>(frame_buffer_.size()) > time_window_) {
        frame_buffer_.pop_front();
    }

    return true;
}

bool TensorReorganizer::is_full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(frame_buffer_.size()) >= time_window_;
}

bool TensorReorganizer::get_tensor(Tensor3D& tensor) const {
    if (!initialized_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    int32_t current_size = static_cast<int32_t>(frame_buffer_.size());
    if (current_size < time_window_) {
        return false;
    }

    tensor = Tensor3D(time_window_, channels_, waves_);

    for (int32_t t = 0; t < time_window_; ++t) {
        const auto& frame = frame_buffer_[t];
        for (int32_t c = 0; c < channels_; ++c) {
            for (int32_t w = 0; w < waves_; ++w) {
                tensor.at(t, c, w) = frame[w];
            }
        }
    }

    return true;
}

bool TensorReorganizer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_buffer_.clear();
    return true;
}

}
