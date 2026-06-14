#include "cmp_epd/trt_engine.h"

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

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
    : trt_runtime_(nullptr)
    , trt_engine_(nullptr)
    , trt_context_(nullptr)
    , input_time_steps_(0)
    , input_channels_(0)
    , input_waves_(0)
    , output_size_(0)
    , gpu_device_id_(0)
    , engine_loaded_(false)
    , use_cuda_(true)
    , buffers_allocated_(false)
    , lstm_state_dirty_(false)
    , pool_ref_count_(0)
    , last_inference_us_(0)
    , avg_inference_ms_(0.0f)
    , inference_count_(0)
    , total_inference_us_(0.0) {
}

TensorRTEngine::~TensorRTEngine() {
    reset();
}

bool TensorRTEngine::init(const EpdConfig& config) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    gpu_device_id_ = config.gpu_device_id;
    use_cuda_ = config.enable_cuda;

#ifdef USE_TENSORRT
    if (use_cuda_) {
        cudaSetDevice(gpu_device_id_);
    }
#endif

    input_time_steps_ = config.time_window;
    input_channels_ = config.polish_heads;
    input_waves_ = config.wave_channels;

    output_size_ = 2;

    lstm_state_.init(2, 2, 1, 128);
    lstm_state_dirty_ = false;

    output_pool_.init(static_cast<size_t>(output_size_));
    pool_ref_count_ = 0;

    last_raw_output_.resize(static_cast<size_t>(output_size_), 0.0f);

    if (!config.trt_engine_path.empty()) {
        if (!load_engine(config.trt_engine_path)) {
            std::cerr << "[TRT] Failed to load engine, running in mock mode" << std::endl;
            engine_loaded_ = false;
        }
    } else {
        engine_loaded_ = false;
    }

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
        static_cast<nvinfer1::IRuntime*>(trt_runtime_)->destroy();
        trt_runtime_ = nullptr;
        return false;
    }

    trt_context_ = static_cast<nvinfer1::IExecutionContext*>(
        static_cast<nvinfer1::ICudaEngine*>(trt_engine_)->createExecutionContext()
    );

    if (!trt_context_) {
        std::cerr << "[TRT] Failed to create execution context" << std::endl;
        static_cast<nvinfer1::ICudaEngine*>(trt_engine_)->destroy();
        trt_engine_ = nullptr;
        static_cast<nvinfer1::IRuntime*>(trt_runtime_)->destroy();
        trt_runtime_ = nullptr;
        return false;
    }

    if (!allocate_static_buffers()) {
        std::cerr << "[TRT] Failed to allocate static buffers" << std::endl;
        return false;
    }

    if (!setup_bindings()) {
        std::cerr << "[TRT] Failed to setup bindings" << std::endl;
        return false;
    }

    engine_loaded_ = true;
    return true;
}

bool TensorRTEngine::allocate_static_buffers() {
    if (buffers_allocated_) {
        free_static_buffers();
    }

    nvinfer1::ICudaEngine* engine = static_cast<nvinfer1::ICudaEngine*>(trt_engine_);
    int32_t num_bindings = engine->getNbBindings();

    device_buffers_.resize(num_bindings, nullptr);
    host_input_buffers_.clear();
    host_output_buffers_.clear();
    buffer_sizes_.resize(num_bindings, 0);
    input_indices_.clear();
    output_indices_.clear();

    for (int32_t i = 0; i < num_bindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        size_t data_size = 1;
        for (int32_t j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] > 0) {
                data_size *= dims.d[j];
            }
        }
        data_size *= sizeof(float);

        buffer_sizes_[i] = data_size;

        if (engine->bindingIsInput(i)) {
            input_indices_.push_back(i);
            void* host_buf = nullptr;
            cudaHostAlloc(&host_buf, data_size, cudaHostAllocWriteCombined);
            host_input_buffers_.push_back(host_buf);
        } else {
            output_indices_.push_back(i);
            void* host_buf = nullptr;
            cudaHostAlloc(&host_buf, data_size, cudaHostAllocDefault);
            host_output_buffers_.push_back(host_buf);
            output_size_ = static_cast<int32_t>(data_size / sizeof(float));
        }

        cudaMalloc(&device_buffers_[i], data_size);
        cudaMemset(device_buffers_[i], 0, data_size);
    }

    buffers_allocated_ = true;
    return true;
}

bool TensorRTEngine::free_static_buffers() {
    for (void*& buf : device_buffers_) {
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
        }
    }
    for (void*& buf : host_input_buffers_) {
        if (buf) {
            cudaFreeHost(buf);
            buf = nullptr;
        }
    }
    for (void*& buf : host_output_buffers_) {
        if (buf) {
            cudaFreeHost(buf);
            buf = nullptr;
        }
    }
    device_buffers_.clear();
    host_input_buffers_.clear();
    host_output_buffers_.clear();
    buffer_sizes_.clear();
    buffers_allocated_ = false;
    return true;
}

bool TensorRTEngine::setup_bindings() {
    return buffers_allocated_ && !device_buffers_.empty();
}

#endif

bool TensorRTEngine::deep_copy_raw_output(const void* src_output, size_t output_bytes) {
    if (output_bytes > last_raw_output_.size() * sizeof(float)) {
        return false;
    }
    std::memcpy(last_raw_output_.data(), src_output, output_bytes);
    return true;
}

bool TensorRTEngine::deep_copy_output(EpdResult& result) {
    size_t expected_size = static_cast<size_t>(output_size_) * sizeof(float);

    float* pool_slot = output_pool_.acquire_snapshot();
    pool_ref_count_++;

    std::memcpy(pool_slot, last_raw_output_.data(), expected_size);

    result.endpoint_probability = pool_slot[0];
    result.thickness_estimate_nm = output_size_ > 1 ? pool_slot[1] : 0.0f;
    result.confidence = pool_slot[0];

    return true;
}

bool TensorRTEngine::infer(const Tensor3D& input, EpdResult& result) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    uint64_t start_time = current_time_us();
    size_t input_data_size = input.size() * sizeof(float);

#ifdef USE_TENSORRT
    if (engine_loaded_ && trt_context_ && buffers_allocated_) {
        nvinfer1::IExecutionContext* ctx = static_cast<nvinfer1::IExecutionContext*>(trt_context_);

        if (!input_indices_.empty() && !host_input_buffers_.empty()) {
            int32_t input_idx = input_indices_[0];
            size_t input_size = buffer_sizes_[input_idx];

            if (input_data_size <= input_size) {
                std::memcpy(host_input_buffers_[0], input.ptr(), input_data_size);

                cudaMemcpyAsync(
                    device_buffers_[input_idx],
                    host_input_buffers_[0],
                    input_data_size,
                    cudaMemcpyHostToDevice,
                    nullptr
                );
            }
        }

        ctx->executeV2(device_buffers_.data());

        if (!output_indices_.empty() && !host_output_buffers_.empty()) {
            int32_t output_idx = output_indices_[0];
            size_t output_size = buffer_sizes_[output_idx];

            cudaMemcpyAsync(
                host_output_buffers_[0],
                device_buffers_[output_idx],
                output_size,
                cudaMemcpyDeviceToHost,
                nullptr
            );

            cudaStreamSynchronize(nullptr);

            deep_copy_raw_output(host_output_buffers_[0], output_size);
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

        float trend = 0.0f;
        if (first_half > 1e-6f) {
            trend = (second_half - first_half) / first_half;
        }

        float prob = std::min(1.0f, std::max(0.0f, 0.5f + trend * 2.0f));
        float thick = 500.0f * (1.0f - prob);

        last_raw_output_[0] = prob;
        last_raw_output_[1] = thick;
    }

    lstm_state_dirty_ = true;

    deep_copy_output(result);

    result.timestamp = static_cast<double>(start_time) / 1e6;
    result.is_endpoint = false;

    uint64_t end_time = current_time_us();
    last_inference_us_ = end_time - start_time;

    inference_count_++;
    total_inference_us_ += static_cast<double>(last_inference_us_);
    avg_inference_ms_ = static_cast<float>(total_inference_us_ / inference_count_ / 1000.0);

    return true;
}

bool TensorRTEngine::reset_lstm_states() {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    lstm_state_.reset();
    lstm_state_dirty_ = false;
    return true;
}

bool TensorRTEngine::reset_output_pool() {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    output_pool_.reset();
    pool_ref_count_ = 0;
    std::memset(last_raw_output_.data(), 0, last_raw_output_.size() * sizeof(float));
    return true;
}

size_t TensorRTEngine::get_total_allocated_bytes() const {
    size_t total = 0;

    for (size_t s : buffer_sizes_) {
        total += s * 2;
    }

    total += lstm_state_.total_bytes();
    total += output_pool_.total_bytes();
    total += last_raw_output_.size() * sizeof(float);

    return total;
}

bool TensorRTEngine::reset() {
    std::lock_guard<std::mutex> lock(inference_mutex_);

#ifdef USE_TENSORRT
    if (buffers_allocated_) {
        free_static_buffers();
    }
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
#endif

    engine_loaded_ = false;
    input_indices_.clear();
    output_indices_.clear();

    lstm_state_.reset();
    lstm_state_dirty_ = false;

    output_pool_.reset();
    pool_ref_count_ = 0;

    std::fill(last_raw_output_.begin(), last_raw_output_.end(), 0.0f);

    return true;
}

}
