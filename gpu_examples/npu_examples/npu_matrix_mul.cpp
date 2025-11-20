#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>


#define MATRIX_SIZE 1024

// CPU matrix multiplication for comparison
void matrix_multiply_cpu(const std::vector<float>& a, const std::vector<float>& b, 
                        std::vector<float>& c, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = sum;
        }
    }
}

// Function to create and run a simple matrix multiplication model on the NPU
void matrix_multiply_npu(const std::vector<float>& a, const std::vector<float>& b, 
                        std::vector<float>& c, int size) {
    try {
        // Initialize OpenVINO Runtime
        ov::Core core;
        
        // Print available devices
        std::cout << "Available OpenVINO devices:" << std::endl;
        for (const auto& device : core.get_available_devices()) {
            std::cout << " - " << device << std::endl;
        }

        // Create a model with matrix multiplication operation
        ov::Shape input_shape{static_cast<size_t>(size), static_cast<size_t>(size)};
        ov::Shape output_shape{static_cast<size_t>(size), static_cast<size_t>(size)};

        // Create model inputs
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        
        // Create matrix multiplication operation
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_a, input_b, false, false);
        
        // Create model with proper OutputVector initialization
        auto outputs = ov::OutputVector{matmul};
        auto params = ov::ParameterVector{input_a, input_b};
        auto model = std::make_shared<ov::Model>(outputs, params);
        
        // Try to compile for NPU, fallback to other devices if not available
        ov::CompiledModel compiled_model;
        try {
            // Try NPU first
            compiled_model = core.compile_model(model, "NPU");
            std::cout << "Using NPU device" << std::endl;
        } catch (const std::exception& e) {
            try {
                // Fall back to GPU
                compiled_model = core.compile_model(model, "GPU");
                std::cout << "Using GPU device (NPU not available: " << e.what() << ")" << std::endl;
            } catch (const std::exception& e) {
                // Fall back to CPU
                compiled_model = core.compile_model(model, "CPU");
                std::cout << "Using CPU device (NPU and GPU not available)" << std::endl;
            }
        }
        
        // Create inference request
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        
        // Create input tensors
        ov::Tensor input_tensor_a(ov::element::f32, input_shape);
        ov::Tensor input_tensor_b(ov::element::f32, input_shape);
        
        // Copy data to tensors
        float* input_data_a = input_tensor_a.data<float>();
        float* input_data_b = input_tensor_b.data<float>();
        
        std::copy(a.begin(), a.end(), input_data_a);
        std::copy(b.begin(), b.end(), input_data_b);
        
        // Set input tensors
        infer_request.set_input_tensor(0, input_tensor_a);
        infer_request.set_input_tensor(1, input_tensor_b);
        
        // Perform inference
        infer_request.infer();
        
        // Get result
        const ov::Tensor& output_tensor = infer_request.get_output_tensor();
        const float* output_data = output_tensor.data<float>();
        
        // Copy result back
        std::copy(output_data, output_data + size * size, c.begin());
        
    } catch (const std::exception& e) {
        std::cerr << "OpenVINO error: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    // Initialize matrices
    std::vector<float> a(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> b(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<float> c_cpu(MATRIX_SIZE * MATRIX_SIZE, 0.0f);
    std::vector<float> c_npu(MATRIX_SIZE * MATRIX_SIZE, 0.0f);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::generate(a.begin(), a.end(), [&]() { return dist(gen); });
    std::generate(b.begin(), b.end(), [&]() { return dist(gen); });
    
    std::cout << "Performing matrix multiplication of size " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    
    // CPU execution timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrix_multiply_cpu(a, b, c_cpu, MATRIX_SIZE);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    std::cout << "CPU execution time: " << cpu_duration.count() << " ms" << std::endl;

    try {
        // NPU execution timing
        auto npu_start = std::chrono::high_resolution_clock::now();
        matrix_multiply_npu(a, b, c_npu, MATRIX_SIZE);
        auto npu_end = std::chrono::high_resolution_clock::now();
        auto npu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(npu_end - npu_start);
        
        std::cout << "NPU execution time: " << npu_duration.count() << " ms" << std::endl;
        std::cout << "Speedup: " << (float)cpu_duration.count() / npu_duration.count() << "x" << std::endl;
        
        // Verify results
        bool correct = true;
        for (size_t i = 0; i < c_cpu.size(); ++i) {
            if (std::abs(c_cpu[i] - c_npu[i]) > 1e-2) {
                std::cout << "Verification failed at index " << i << ": CPU=" << c_cpu[i] 
                          << ", NPU=" << c_npu[i] << std::endl;
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "Verification: PASSED" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during NPU execution: " << e.what() << std::endl;
    }


    return 0;
} 