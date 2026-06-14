#include "cmp_epd/sg_filter.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace cmp_epd {

namespace {

void matrix_multiply(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B,
    std::vector<std::vector<float>>& C
) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t p = B.size();
    C.assign(m, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < p; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void matrix_transpose(
    const std::vector<std::vector<float>>& A,
    std::vector<std::vector<float>>& AT
) {
    size_t m = A.size();
    size_t n = A[0].size();
    AT.assign(n, std::vector<float>(m, 0.0f));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            AT[j][i] = A[i][j];
        }
    }
}

bool matrix_inverse(
    const std::vector<std::vector<float>>& A,
    std::vector<std::vector<float>>& inv
) {
    size_t n = A.size();
    if (n == 0 || A[0].size() != n) {
        return false;
    }

    std::vector<std::vector<float>> aug(n, std::vector<float>(2 * n, 0.0f));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = A[i][j];
        }
        aug[i][i + n] = 1.0f;
    }

    for (size_t col = 0; col < n; ++col) {
        size_t pivot_row = col;
        float max_val = std::abs(aug[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            if (std::abs(aug[row][col]) > max_val) {
                max_val = std::abs(aug[row][col]);
                pivot_row = row;
            }
        }

        if (max_val < 1e-10f) {
            return false;
        }

        if (pivot_row != col) {
            std::swap(aug[col], aug[pivot_row]);
        }

        float pivot = aug[col][col];
        for (size_t j = col; j < 2 * n; ++j) {
            aug[col][j] /= pivot;
        }

        for (size_t row = 0; row < n; ++row) {
            if (row != col && std::abs(aug[row][col]) > 1e-10f) {
                float factor = aug[row][col];
                for (size_t j = col; j < 2 * n; ++j) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    inv.assign(n, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inv[i][j] = aug[i][j + n];
        }
    }

    return true;
}

}

SavitzkyGolayFilter::SavitzkyGolayFilter()
    : window_size_(0)
    , poly_order_(0)
    , half_window_(0)
    , wave_channels_(0)
    , initialized_(false)
    , pad_edges_(true) {
}

bool SavitzkyGolayFilter::init(int32_t window_size, int32_t poly_order, int32_t wave_channels) {
    if (window_size < 5 || window_size % 2 == 0) {
        return false;
    }
    if (poly_order < 1 || poly_order >= window_size) {
        return false;
    }
    if (poly_order > 4) {
        return false;
    }
    if (wave_channels <= 0) {
        return false;
    }

    window_size_ = window_size;
    poly_order_ = poly_order;
    half_window_ = window_size_ / 2;
    wave_channels_ = wave_channels;
    initialized_ = true;

    compute_weights();
    return true;
}

void SavitzkyGolayFilter::compute_weights() {
    int32_t n = window_size_;
    int32_t m = poly_order_ + 1;

    std::vector<std::vector<float>> J(n, std::vector<float>(m, 0.0f));
    for (int32_t i = 0; i < n; ++i) {
        float x = static_cast<float>(i - half_window_);
        float val = 1.0f;
        for (int32_t j = 0; j < m; ++j) {
            J[i][j] = val;
            val *= x;
        }
    }

    std::vector<std::vector<float>> JT;
    matrix_transpose(J, JT);

    std::vector<std::vector<float>> JTJ;
    matrix_multiply(JT, J, JTJ);

    std::vector<std::vector<float>> JTJ_inv;
    if (!matrix_inverse(JTJ, JTJ_inv)) {
        for (int32_t i = 0; i < n; ++i) {
            weights_.push_back(1.0f / static_cast<float>(n));
        }
        return;
    }

    std::vector<std::vector<float>> pseudo_inv;
    matrix_multiply(JTJ_inv, JT, pseudo_inv);

    weights_.resize(n);
    for (int32_t i = 0; i < n; ++i) {
        weights_[i] = pseudo_inv[0][i];
    }
}

bool SavitzkyGolayFilter::filter(const std::vector<float>& input, std::vector<float>& output) {
    if (!initialized_) {
        return false;
    }
    if (static_cast<int32_t>(input.size()) != wave_channels_) {
        return false;
    }

    output.resize(wave_channels_);

    for (int32_t i = 0; i < wave_channels_; ++i) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int32_t j = -half_window_; j <= half_window_; ++j) {
            int32_t idx = i + j;

            if (idx < 0) {
                if (pad_edges_) {
                    idx = 0;
                } else {
                    continue;
                }
            } else if (idx >= wave_channels_) {
                if (pad_edges_) {
                    idx = wave_channels_ - 1;
                } else {
                    continue;
                }
            }

            float w = weights_[j + half_window_];
            sum += w * input[idx];
            weight_sum += w;
        }

        if (std::abs(weight_sum) > 1e-6f) {
            output[i] = sum / weight_sum;
        } else {
            output[i] = input[i];
        }
    }

    return true;
}

bool SavitzkyGolayFilter::reset() {
    initialized_ = false;
    window_size_ = 0;
    poly_order_ = 0;
    half_window_ = 0;
    wave_channels_ = 0;
    weights_.clear();
    return true;
}

}
