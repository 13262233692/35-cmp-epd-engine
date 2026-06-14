#include "cmp_epd/trt_engine.h"

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace cmp_epd {

#ifdef USE_TENSORRT

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
} g_logger;

#endif

TensorRTEngine::TensorRTEngine()
    : cuda_runtime_(nullptr)
    , trt_runtime_(nullptr)
    , trt_engine_(nullptr)
    , trt_context_(nullptr)
    , input_time_steps_(0)
    , input_channels_(0)
    , input_waves_(0)
    , output_size_(0)
    , gpu_device_id_(0)
    , engine_loaded_(false)
    , use_cuda_(true)
    , last_inference_us_(0)
    , avg_inference_ms_(0.0f)
    , inference_count_(0)
    , total_inference_us_(0.0) {
}

TensorRTEngine::~TensorRTEngine() {
    reset();
}

bool TensorRTEngine::init(const EpdConfig& config) {
    gpu_device_id_ = config.gpu_device_id;
    use_cuda_ = config.enable_cuda;

#ifdef USE_TENSORRT
    cudaSetDevice(gpu_device_id_);

    input_time_steps_ = config.time_window;
    input_channels_ = config.polish_heads;
    input_waves_ = config.wave_channels;

    if (!config.trt_engine_path.empty()) {
        if (!load_engine(config.trt_engine_path)) {
            std::cerr << "[TRT] Failed to load engine: " << config.trt_engine_path << std::endl;
            return false;
        }
    } else {
        std::cerr << "[TRT] No engine path provided, running in mock mode" << std::endl;
        output_size_ = 2;
        engine_loaded_ = false;
    }
#else
    input_time_steps_ = config.time_window;
    input_channels_ = config.polish_heads;
    input_waves_ = config.wave_channels;
    output_size_ = 2;
    engine_loaded_ = false;
#endif

    return true;
}

#ifdef USE_TENSORRT

bool TensorRTEngine::load_engine(const std::string& engine_path) {
    std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
    if (!engine_file.is_open()) {
        std::cerr << "[TRT] Cannot open engine file: " << engine_path << std::endl;
        return false;
    }

    size_t engine_size = static_cast<size_t>(engine_file.tellg());
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    trt_runtime_ = nvinfer1::createInferRuntime(g_logger);
    if (!trt_runtime_) {
        std::cerr << "[TRT] Failed to create runtime" << std::endl;
        return false;
    }

    trt_engine_ = static_cast<nvinfer1::ICudaEngine*>(
        static_cast<nvinfer1::IRuntime*>(trt_runtime_)->deserializeCudaEngine(
            engine_data.data(), engine_size, nullptr
        )
    );

    if (!trt_engine_) {
        std::cerr << "[TRT] Failed to deserialize engine" << std::endl;
        return false;
    }

    trt_context_ = static_cast<nvinfer1::IExecutionContext*>(
        static_cast<nvinfer1::ICudaEngine*>(trt_engine_)->createExecutionContext()
    );

    if (!trt_context_) {
        std::cerr << "[TRT] Failed to create execution context" << std::endl;
        return false;
    }

    if (!allocate_buffers()) {
        std::cerr << "[TRT] Failed to allocate buffers" << std::endl;
        return false;
    }

    if (!setup_bindings()) {
        std::cerr << "[TRT] Failed to setup bindings" << std::endl;
        return false;
    }

    engine_loaded_ = true;
    return true;
}

bool TensorRTEngine::allocate_buffers() {
    nvinfer1::ICudaEngine* engine = static_cast<nvinfer1::ICudaEngine*>(trt_engine_);
    int32_t num_bindings = engine->getNbBindings();

    device_buffers_.resize(num_bindings, nullptr);
    host_buffers_.resize(num_bindings, nullptr);
    buffer_sizes_.resize(num_bindings, 0);

    for (int32_t i = 0; i < num_bindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        size_t data_size = 1;
        for (int32_t j = 0; j < dims.nbDims; ++j) {
            data_size *= dims.d[j];
        }
        data_size *= (dtype == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(float));

        buffer_sizes_[i] = data_size;

        cudaMalloc(&device_buffers_[i], data_size);
        host_buffers_[i] = malloc(data_size);

        if (engine->bindingIsInput(i)) {
            input_indices_.push_back(i);
        } else {
            output_indices_.push_back(i);
            output_size_ = static_cast<int32_t>(data_size / sizeof(float));
        }
    }

    return true;
}

bool TensorRTEngine::free_buffers() {
    for (void*& buf : device_buffers_) {
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
        }
    }
    for (void*& buf : host_buffers_) {
        if (buf) {
            free(buf);
            buf = nullptr;
        }
    }
    device_buffers_.clear();
    host_buffers_.clear();
    buffer_sizes_.clear();
    return true;
}

bool TensorRTEngine::setup_bindings() {
    return !device_buffers_.empty();
}

#endif

bool TensorRTEngine::infer(const Tensor3D& input, EpdResult& result) {
    uint64_t start_time = current_time_us();

#ifdef USE_TENSORRT
    if (engine_loaded_ && trt_context_) {
        nvinfer1::IExecutionContext* ctx = static_cast<nvinfer1::IExecutionContext*>(trt_context_);

        if (!input_indices_.empty()) {
            int32_t input_idx = input_indices_[0];
            size_t input_size = buffer_sizes_[input_idx];
            size_t data_size = input.size() * sizeof(float);

            if (data_size == input_size) {
                cudaMemcpy(
                    device_buffers_[input_idx],
                    input.ptr(),
                    input_size,
                    cudaMemcpyHostToDevice
                );
            }
        }

        ctx->executeV2(device_buffers_.data());

        if (!output_indices_.empty()) {
            int32_t output_idx = output_indices_[0];
            size_t output_size = buffer_sizes_[output_idx];

            cudaMemcpy(
                host_buffers_[output_idx],
                device_buffers_[output_idx],
                output_size,
                cudaMemcpyDeviceToHost
            );

            float* output_data = static_cast<float*>(host_buffers_[output_idx]);
            result.endpoint_probability = output_data[0];
            result.thickness_estimate_nm = output_size > 1 ? output_data[1] : 0.0f;
            result.confidence = output_data[0];
        }
    } else
#endif
    {
        float sum_intensity = 0.0f;
        for (int32_t t = 0; t < input.time_steps; ++t) {
            for (int32_t w = 0; w < input.waves; ++w) {
                sum_intensity += std::abs(input.at(t, 0, w));
            }
        }
        float avg_intensity = sum_intensity / static_cast<float>(input.time_steps * input.waves);

        float trend = 0.0f;
        int32_t half = input.time_steps / 2;
        float first_half = 0.0f, second_half = 0.0f;

        for (int32_t t = 0; t < half; ++t) {
            for (int32_t w = 0; w < input.waves; ++w) {
                first_half += std::abs(input.at(t, 0, w));
            }
        }
        for (int32_t t = half; t < input.time_steps; ++t) {
            for (int32_t w = 0; w < input.waves; ++w) {
                second_half += std::abs(input.at(t, 0, w));
            }
        }

        if (first_half > 1e-6f) {
            trend = (second_half - first_half) / first_half;
        }

        result.endpoint_probability = std::min(1.0f, std::max(0.0f, 0.5f + trend * 2.0f));
        result.thickness_estimate_nm = 500.0f * (1.0f - result.endpoint_probability);
        result.confidence = 0.6f + 0.4f * avg_intensity;
    }

    result.timestamp = static_cast<double>(start_time) / 1e6;
    result.is_endpoint = false;

    uint64_t end_time = current_time_us();
    last_inference_us_ = end_time - start_time;

    inference_count_++;
    total_inference_us_ += static_cast<double>(last_inference_us_);
    avg_inference_ms_ = static_cast<float>(total_inference_us_ / inference_count_ / 1000.0);

    return true;
}

bool TensorRTEngine::reset() {
#ifdef USE_TENSORRT
    if (trt_context_) {
        static_cast<nvinfer1::IExecutionContext*>(trt_context_)->destroy();
        trt_context_ = nullptr;
    }
    if (trt_engine_) {
        static_cast<nvinfer1::ICudaEngine*>(trt_engine_)->destroy();
        trt_engine_ = nullptr;
    }
    if (trt_runtime_) {
        static_cast<nvinfer1::IRuntime*>(trt_runtime_)->destroy();
        trt_runtime_ = nullptr;
    }
    free_buffers();
#endif

    engine_loaded_ = false;
    input_indices_.clear();
    output_indices_.clear();

    return true;
}

}
