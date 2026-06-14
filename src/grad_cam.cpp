#include "cmp_epd/grad_cam.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace cmp_epd {

namespace {

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

void jet_colormap(float val, uint8_t& r, uint8_t& g, uint8_t& b) {
    val = std::max(0.0f, std::min(1.0f, val));

    if (val < 0.125f) {
        r = 0;
        g = 0;
        b = static_cast<uint8_t>(128 + 127 * (val / 0.125f));
    } else if (val < 0.375f) {
        r = 0;
        g = static_cast<uint8_t>(255 * (val - 0.125f) / 0.25f);
        b = 255;
    } else if (val < 0.625f) {
        r = static_cast<uint8_t>(255 * (val - 0.375f) / 0.25f);
        g = 255;
        b = static_cast<uint8_t>(255 * (1.0f - (val - 0.375f) / 0.25f));
    } else if (val < 0.875f) {
        r = 255;
        g = static_cast<uint8_t>(255 * (1.0f - (val - 0.625f) / 0.25f));
        b = 0;
    } else {
        r = static_cast<uint8_t>(255);
        g = 0;
        b = 0;
    }
}

}

GradCam::GradCam()
    : initialized_(false)
    , conv_channels_(0)
    , conv_spatial_size_(0)
    , weights_cached_(false)
    , trigger_count_(0) {
}

bool GradCam::init(const GradCamConfig& config, int32_t conv_channels, int32_t conv_spatial_size) {
    if (conv_channels <= 0 || conv_spatial_size <= 0) {
        return false;
    }
    if (config.heatmap_width <= 0 || config.heatmap_height <= 0) {
        return false;
    }

    config_ = config;
    conv_channels_ = conv_channels;
    conv_spatial_size_ = conv_spatial_size;

    cached_weights_.resize(static_cast<size_t>(conv_channels_), 0.0f);
    weights_cached_ = false;
    trigger_count_ = 0;

    initialized_ = true;
    return true;
}

bool GradCam::compute(const std::vector<float>& conv_features,
                      int32_t channels,
                      int32_t spatial_size,
                      GradCamResult& result) {
    if (!initialized_) {
        return false;
    }
    if (static_cast<int32_t>(conv_features.size()) != channels * spatial_size) {
        return false;
    }

    result.allocate(config_.heatmap_width, config_.heatmap_height);
    result.wafer_diameter_mm = config_.wafer_diameter_mm;
    result.pixel_per_mm = static_cast<float>(config_.heatmap_width) / config_.wafer_diameter_mm;

    result.conv_channels = channels;
    result.conv_spatial_size = spatial_size;
    result.conv_feature_map = conv_features;
    result.grad_weights = cached_weights_;

    std::vector<float> cam(static_cast<size_t>(spatial_size), 0.0f);

    for (int32_t c = 0; c < channels; ++c) {
        float w = cached_weights_[c];
        const float* feat = conv_features.data() + c * spatial_size;
        for (int32_t s = 0; s < spatial_size; ++s) {
            cam[s] += w * feat[s];
        }
    }

    for (auto& v : cam) {
        v = std::max(0.0f, v);
    }

    int32_t cam_side = static_cast<int32_t>(std::sqrt(spatial_size));
    if (cam_side * cam_side != spatial_size) {
        cam_side = spatial_size;
    }

    std::vector<float> cam_1d = cam;
    int32_t in_size = (cam_side > 0) ? cam_side : spatial_size;

    upsample_bilinear(cam_1d, in_size, 1, result.heatmap,
                      config_.heatmap_width, config_.heatmap_height);

    normalize_heatmap(result);
    mask_to_wafer_circle(result);
    detect_critical_zones(result);

    result.valid = true;
    result.timestamp = static_cast<double>(current_time_us()) / 1e6;

    trigger_count_++;

    return true;
}

bool GradCam::compute_with_approx(const Tensor3D& input_tensor,
                                  float endpoint_prob,
                                  GradCamResult& result) {
    if (!initialized_) {
        return false;
    }

    int32_t spatial_size = input_tensor.waves;
    int32_t channels = input_tensor.channels * 8;

    std::vector<float> approx_features(static_cast<size_t>(channels) * spatial_size, 0.0f);

    for (int32_t c = 0; c < input_tensor.channels; ++c) {
        for (int32_t w = 0; w < input_tensor.waves; ++w) {
            float val = 0.0f;
            for (int32_t t = 0; t < input_tensor.time_steps; ++t) {
                val += std::abs(input_tensor.at(t, c, w));
            }
            val /= static_cast<float>(input_tensor.time_steps);
            approx_features[c * spatial_size + w] = val;
        }
    }

    for (int32_t c = input_tensor.channels; c < channels; ++c) {
        int32_t src_c = c % input_tensor.channels;
        float scale = 1.0f + 0.1f * static_cast<float>(c - input_tensor.channels);
        for (int32_t w = 0; w < spatial_size; ++w) {
            float base = approx_features[src_c * spatial_size + w];
            float wave = std::sin(static_cast<float>(w) * 0.02f * static_cast<float>(c) + c);
            approx_features[c * spatial_size + w] = base * scale * (1.0f + 0.1f * wave);
        }
    }

    compute_approx_grad_weights(input_tensor, endpoint_prob, cached_weights_);
    weights_cached_ = true;

    if (channels < conv_channels_) {
        int32_t pad_channels = conv_channels_;
        std::vector<float> padded_features(static_cast<size_t>(pad_channels) * spatial_size, 0.0f);

        int32_t copy_channels = std::min(channels, pad_channels);
        for (int32_t c = 0; c < copy_channels; ++c) {
            std::memcpy(padded_features.data() + c * spatial_size,
                       approx_features.data() + c * spatial_size,
                       spatial_size * sizeof(float));
        }

        for (int32_t c = copy_channels; c < pad_channels; ++c) {
            int32_t src = c % copy_channels;
            std::memcpy(padded_features.data() + c * spatial_size,
                       approx_features.data() + src * spatial_size,
                       spatial_size * sizeof(float));
        }

        return compute(padded_features, pad_channels, spatial_size, result);
    }

    return compute(approx_features, channels, spatial_size, result);
}

bool GradCam::compute_approx_grad_weights(const Tensor3D& input_tensor,
                                          float endpoint_prob,
                                          std::vector<float>& weights) {
    int32_t num_weights = static_cast<int32_t>(weights.size());
    if (num_weights <= 0) return false;

    int32_t half = input_tensor.time_steps / 2;

    for (int32_t i = 0; i < num_weights; ++i) {
        float wave_pos = static_cast<float>(i % input_tensor.waves)
            / static_cast<float>(input_tensor.waves);
        float time_trend = 0.0f;

        if (half > 0) {
            float first_half = 0.0f;
            float second_half = 0.0f;

            int32_t wave_idx = i % input_tensor.waves;
            int32_t ch_idx = (i / input_tensor.waves) % input_tensor.channels;

            for (int32_t t = 0; t < half; ++t) {
                first_half += std::abs(input_tensor.at(t, ch_idx, wave_idx));
            }
            for (int32_t t = half; t < input_tensor.time_steps; ++t) {
                second_half += std::abs(input_tensor.at(t, ch_idx, wave_idx));
            }

            first_half /= static_cast<float>(half);
            second_half /= static_cast<float>(input_tensor.time_steps - half);

            if (first_half > 1e-6f) {
                time_trend = (second_half - first_half) / first_half;
            }
        }

        float spectral_weight = std::sin(wave_pos * 3.14159f + i * 0.1f);
        float prob_weight = endpoint_prob * 0.8f + 0.2f;

        weights[i] = (0.5f + 0.5f * time_trend) * (0.7f + 0.3f * spectral_weight) * prob_weight;
        weights[i] = std::max(0.0f, weights[i]);
    }

    float sum = 0.0f;
    for (float w : weights) sum += w;
    if (sum > 1e-6f) {
        for (float& w : weights) w /= sum;
    }

    return true;
}

bool GradCam::upsample_bilinear(const std::vector<float>& input,
                                int32_t in_w, int32_t in_h,
                                std::vector<std::vector<float>>& output,
                                int32_t out_w, int32_t out_h) {
    if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) {
        return false;
    }

    output.assign(out_h, std::vector<float>(out_w, 0.0f));

    float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
    float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);

    for (int32_t y = 0; y < out_h; ++y) {
        for (int32_t x = 0; x < out_w; ++x) {
            float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

            src_x = std::max(0.0f, std::min(static_cast<float>(in_w - 1), src_x));
            src_y = std::max(0.0f, std::min(static_cast<float>(in_h - 1), src_y));

            int32_t x0 = static_cast<int32_t>(std::floor(src_x));
            int32_t y0 = static_cast<int32_t>(std::floor(src_y));
            int32_t x1 = std::min(x0 + 1, in_w - 1);
            int32_t y1 = std::min(y0 + 1, in_h - 1);

            float fx = src_x - static_cast<float>(x0);
            float fy = src_y - static_cast<float>(y0);

            int32_t idx00 = y0 * in_w + x0;
            int32_t idx10 = y0 * in_w + x1;
            int32_t idx01 = y1 * in_w + x0;
            int32_t idx11 = y1 * in_w + x1;

            idx00 = std::max(0, std::min(static_cast<int32_t>(input.size()) - 1, idx00));
            idx10 = std::max(0, std::min(static_cast<int32_t>(input.size()) - 1, idx10));
            idx01 = std::max(0, std::min(static_cast<int32_t>(input.size()) - 1, idx01));
            idx11 = std::max(0, std::min(static_cast<int32_t>(input.size()) - 1, idx11));

            float v00 = input[idx00];
            float v10 = input[idx10];
            float v01 = input[idx01];
            float v11 = input[idx11];

            float top = lerp(v00, v10, fx);
            float bottom = lerp(v01, v11, fx);
            float val = lerp(top, bottom, fy);

            output[y][x] = val;
        }
    }

    return true;
}

bool GradCam::normalize_heatmap(GradCamResult& result) {
    if (result.heatmap.empty()) return false;

    float min_val = 1e9f;
    float max_val = -1e9f;
    float sum = 0.0f;
    int32_t count = 0;

    for (const auto& row : result.heatmap) {
        for (float v : row) {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            sum += v;
            count++;
        }
    }

    result.min_activation = min_val;
    result.max_activation = max_val;
    result.avg_activation = count > 0 ? sum / count : 0.0f;

    float range = max_val - min_val;
    if (range > 1e-6f) {
        for (auto& row : result.heatmap) {
            for (float& v : row) {
                v = (v - min_val) / range;
            }
        }
    }

    return true;
}

bool GradCam::mask_to_wafer_circle(GradCamResult& result) {
    if (result.heatmap.empty()) return false;

    int32_t w = result.heatmap_width;
    int32_t h = result.heatmap_height;

    float cx = static_cast<float>(w) / 2.0f;
    float cy = static_cast<float>(h) / 2.0f;
    float radius = std::min(cx, cy) * 0.95f;

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist > radius) {
                result.heatmap[y][x] = 0.0f;
            } else if (dist > radius * 0.9f) {
                float edge = (dist - radius * 0.9f) / (radius * 0.1f);
                result.heatmap[y][x] *= (1.0f - edge);
            }
        }
    }

    return true;
}

bool GradCam::detect_critical_zones(GradCamResult& result) {
    if (result.heatmap.empty()) return false;

    int32_t w = result.heatmap_width;
    int32_t h = result.heatmap_height;
    float cx = static_cast<float>(w) / 2.0f;
    float cy = static_cast<float>(h) / 2.0f;
    float radius_mm = config_.wafer_diameter_mm / 2.0f;
    float px_per_mm = static_cast<float>(w) / config_.wafer_diameter_mm;

    std::vector<WaferZone> zones;

    const int32_t ring_count = 3;
    const int32_t sector_count = 6;

    for (int32_t ring = 0; ring < ring_count; ++ring) {
        float inner_r = radius_mm * (0.2f + static_cast<float>(ring) * 0.25f);
        float outer_r = radius_mm * (0.2f + static_cast<float>(ring + 1) * 0.25f);

        for (int32_t sec = 0; sec < sector_count; ++sec) {
            float angle_start = static_cast<float>(sec) / sector_count * 2.0f * 3.14159f;
            float angle_end = static_cast<float>(sec + 1) / sector_count * 2.0f * 3.14159f;

            WaferZone zone;
            zone.name = "Ring" + std::to_string(ring) + "_Sec" + std::to_string(sec);
            zone.avg_heat = 0.0f;
            zone.max_heat = 0.0f;
            zone.is_critical_zone = false;

            int32_t count = 0;
            float sum = 0.0f;

            for (int32_t y = 0; y < h; ++y) {
                for (int32_t x = 0; x < w; ++x) {
                    float dx = static_cast<float>(x) - cx;
                    float dy = static_cast<float>(y) - cy;
                    float dist_mm = std::sqrt(dx * dx + dy * dy) / px_per_mm;
                    float angle = std::atan2(dy, dx);
                    if (angle < 0) angle += 2.0f * 3.14159f;

                    if (dist_mm >= inner_r && dist_mm < outer_r &&
                        angle >= angle_start && angle < angle_end) {
                        float val = result.heatmap[y][x];
                        sum += val;
                        count++;
                        if (val > zone.max_heat) zone.max_heat = val;
                    }
                }
            }

            if (count > 0) {
                zone.avg_heat = sum / count;
                zone.center_x_mm = (inner_r + outer_r) / 2.0f
                    * std::cos((angle_start + angle_end) / 2.0f);
                zone.center_y_mm = (inner_r + outer_r) / 2.0f
                    * std::sin((angle_start + angle_end) / 2.0f);
                zone.radius_mm = (outer_r - inner_r) / 2.0f;
                zones.push_back(zone);
            }
        }
    }

    WaferZone center;
    center.name = "Center";
    center.center_x_mm = 0.0f;
    center.center_y_mm = 0.0f;
    center.radius_mm = radius_mm * 0.2f;
    center.avg_heat = 0.0f;
    center.max_heat = 0.0f;
    center.is_critical_zone = false;

    int32_t center_count = 0;
    float center_sum = 0.0f;
    float center_radius_px = center.radius_mm * px_per_mm;

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist <= center_radius_px) {
                float val = result.heatmap[y][x];
                center_sum += val;
                center_count++;
                if (val > center.max_heat) center.max_heat = val;
            }
        }
    }
    if (center_count > 0) {
        center.avg_heat = center_sum / center_count;
        zones.push_back(center);
    }

    float threshold = 0.75f;
    for (auto& zone : zones) {
        if (zone.avg_heat >= threshold) {
            zone.is_critical_zone = true;
            result.critical_zones.push_back(zone);
        }
    }

    if (!zones.empty()) {
        result.hottest_zone = zones[0];
        for (const auto& zone : zones) {
            if (zone.avg_heat > result.hottest_zone.avg_heat) {
                result.hottest_zone = zone;
            }
        }
    }

    float contour_thresh = 0.6f;
    find_contour(result, contour_thresh, result.contour_points);

    return true;
}

bool GradCam::find_contour(const GradCamResult& result,
                           float threshold,
                           std::vector<std::pair<int32_t, int32_t>>& contour) {
    if (result.heatmap.empty()) return false;

    contour.clear();

    int32_t w = result.heatmap_width;
    int32_t h = result.heatmap_height;

    const int32_t step = 2;
    for (int32_t y = 1; y < h - 1; y += step) {
        for (int32_t x = 1; x < w - 1; x += step) {
            float center = result.heatmap[y][x];
            if (center >= threshold) {
                bool is_edge = false;
                for (int dy = -1; dy <= 1 && !is_edge; ++dy) {
                    for (int dx = -1; dx <= 1 && !is_edge; ++dx) {
                        if (result.heatmap[y + dy][x + dx] < threshold) {
                            is_edge = true;
                        }
                    }
                }
                if (is_edge) {
                    contour.push_back({x, y});
                }
            }
        }
    }

    return true;
}

bool GradCam::apply_jet_colormap(const GradCamResult& result,
                                  std::vector<HeatmapPixel>& colormap) {
    if (!result.valid || result.heatmap.empty()) {
        return false;
    }

    int32_t w = result.heatmap_width;
    int32_t h = result.heatmap_height;

    colormap.resize(static_cast<size_t>(w) * h);

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
            float val = result.heatmap[y][x];
            HeatmapPixel& px = colormap[y * w + x];
            px.intensity = val;
            jet_colormap(val, px.r, px.g, px.b);
        }
    }

    return true;
}

void GradCam::reset() {
    trigger_count_ = 0;
    weights_cached_ = false;
    std::fill(cached_weights_.begin(), cached_weights_.end(), 0.0f);
}

}
