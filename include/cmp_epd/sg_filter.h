#pragma once

#include "common.h"
#include <vector>
#include <cstdint>

namespace cmp_epd {

class SavitzkyGolayFilter {
public:
    SavitzkyGolayFilter();
    ~SavitzkyGolayFilter() = default;

    bool init(int32_t window_size, int32_t poly_order, int32_t wave_channels);

    bool filter(const std::vector<float>& input, std::vector<float>& output);

    bool reset();

    int32_t window_size() const { return window_size_; }
    int32_t poly_order() const { return poly_order_; }
    int32_t wave_channels() const { return wave_channels_; }
    bool is_initialized() const { return initialized_; }

    void set_edge_mode(bool pad_edges) { pad_edges_ = pad_edges; }

private:
    void compute_weights();
    float gram_polynomial(int32_t i, int32_t m, int32_t k, int32_t s) const;
    float gen_fact(int32_t a, int32_t b) const;

    int32_t window_size_;
    int32_t poly_order_;
    int32_t half_window_;
    int32_t wave_channels_;
    bool initialized_;
    bool pad_edges_;

    std::vector<float> weights_;
};

}
