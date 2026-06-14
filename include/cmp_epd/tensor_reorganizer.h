#pragma once

#include "common.h"
#include "sg_filter.h"
#include <vector>
#include <deque>
#include <mutex>
#include <cstdint>
#include <memory>

namespace cmp_epd {

class TensorReorganizer {
public:
    TensorReorganizer();
    ~TensorReorganizer() = default;

    bool init(const EpdConfig& config);

    bool push_frame(const SpectrumFrame& frame);

    bool get_tensor(Tensor3D& tensor) const;

    bool is_full() const;

    bool reset();

    int32_t time_window() const { return time_window_; }
    int32_t channels() const { return channels_; }
    int32_t waves() const { return waves_; }
    int32_t current_size() const { return static_cast<int32_t>(frame_buffer_.size()); }

    void set_normalization(bool enable) { normalize_ = enable; }
    bool normalization_enabled() const { return normalize_; }

private:
    void normalize_spectrum(std::vector<float>& spectrum) const;
    bool validate_frame(const SpectrumFrame& frame) const;

    int32_t time_window_;
    int32_t channels_;
    int32_t waves_;
    bool initialized_;
    bool normalize_;

    std::unique_ptr<SavitzkyGolayFilter> sg_filter_;
    std::deque<std::vector<float>> frame_buffer_;

    mutable std::mutex mutex_;
};

}
