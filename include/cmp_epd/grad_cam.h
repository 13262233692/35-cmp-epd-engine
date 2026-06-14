#pragma once

#include "common.h"
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace cmp_epd {

struct GradCamConfig {
    float trigger_threshold = 0.90f;
    int32_t target_conv_layer_index = -1;
    std::string target_conv_layer_name;
    int32_t heatmap_width = 200;
    int32_t heatmap_height = 200;
    float wafer_diameter_mm = 300.0f;
    bool apply_colormap = true;
    int32_t min_contour_area_pct = 5;
    bool enable_high_quality_upsample = true;
};

struct HeatmapPixel {
    float intensity;
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct WaferZone {
    std::string name;
    float center_x_mm;
    float center_y_mm;
    float radius_mm;
    float avg_heat;
    float max_heat;
    bool is_critical_zone;
};

struct GradCamResult {
    bool valid;
    double timestamp;

    std::vector<std::vector<float>> heatmap;
    int32_t heatmap_width;
    int32_t heatmap_height;

    float wafer_diameter_mm;
    float pixel_per_mm;

    float max_activation;
    float avg_activation;
    float min_activation;

    std::vector<WaferZone> critical_zones;
    WaferZone hottest_zone;

    std::vector<std::pair<int32_t, int32_t>> contour_points;

    std::vector<float> conv_feature_map;
    int32_t conv_channels;
    int32_t conv_spatial_size;

    std::vector<float> grad_weights;

    GradCamResult()
        : valid(false)
        , timestamp(0.0)
        , heatmap_width(0)
        , heatmap_height(0)
        , wafer_diameter_mm(300.0f)
        , pixel_per_mm(0.0f)
        , max_activation(0.0f)
        , avg_activation(0.0f)
        , min_activation(0.0f)
        , conv_channels(0)
        , conv_spatial_size(0) {}

    void allocate(int32_t w, int32_t h) {
        heatmap_width = w;
        heatmap_height = h;
        heatmap.assign(h, std::vector<float>(w, 0.0f));
    }

    size_t total_bytes() const {
        size_t total = 0;
        total += heatmap.size() * sizeof(std::vector<float>);
        for (const auto& row : heatmap) {
            total += row.size() * sizeof(float);
        }
        total += critical_zones.size() * sizeof(WaferZone);
        total += contour_points.size() * sizeof(std::pair<int32_t, int32_t>);
        total += conv_feature_map.size() * sizeof(float);
        total += grad_weights.size() * sizeof(float);
        return total;
    }
};

class GradCam {
public:
    GradCam();
    ~GradCam() = default;

    bool init(const GradCamConfig& config, int32_t conv_channels, int32_t conv_spatial_size);

    bool compute(const std::vector<float>& conv_features,
                 int32_t channels,
                 int32_t spatial_size,
                 GradCamResult& result);

    bool compute_with_approx(const Tensor3D& input_tensor,
                             float endpoint_prob,
                             GradCamResult& result);

    bool detect_critical_zones(GradCamResult& result);

    bool apply_jet_colormap(const GradCamResult& result,
                            std::vector<HeatmapPixel>& colormap);

    bool is_triggered(float endpoint_prob) const {
        return endpoint_prob >= config_.trigger_threshold;
    }

    void reset();

    const GradCamConfig& config() const { return config_; }
    bool is_initialized() const { return initialized_; }

    int32_t conv_channels() const { return conv_channels_; }
    int32_t conv_spatial_size() const { return conv_spatial_size_; }

private:
    bool upsample_bilinear(const std::vector<float>& input,
                           int32_t in_w, int32_t in_h,
                           std::vector<std::vector<float>>& output,
                           int32_t out_w, int32_t out_h);

    bool compute_approx_grad_weights(const Tensor3D& input_tensor,
                                     float endpoint_prob,
                                     std::vector<float>& weights);

    bool normalize_heatmap(GradCamResult& result);

    bool mask_to_wafer_circle(GradCamResult& result);

    bool find_contour(const GradCamResult& result,
                      float threshold,
                      std::vector<std::pair<int32_t, int32_t>>& contour);

    GradCamConfig config_;
    bool initialized_;

    int32_t conv_channels_;
    int32_t conv_spatial_size_;

    std::vector<float> cached_weights_;
    bool weights_cached_;

    int32_t trigger_count_;
};

}
