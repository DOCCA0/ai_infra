#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "Initializing ONNX Runtime..." << std::endl;

    // 1. Initialize Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNetInference");

    // 2. Session Options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 3. Load Model
    // Assuming the executable is run from the same directory as the model or we provide full path
    // For this environment, we'll assume the model is in the current directory
    const char* model_path = "./simple_resnet.onnx";
    
    std::cout << "Loading model from " << model_path << "..." << std::endl;
    try {
        Ort::Session session(env, model_path, session_options);

        // 4. Prepare Inputs
        // The model input is [1, 3, 32, 32] based on resnet_export.py
        std::vector<int64_t> input_shape = {1, 3, 32, 32};
        size_t input_tensor_size = 1 * 3 * 32 * 32;
        std::vector<float> input_tensor_values(input_tensor_size);
        
        // Initialize with some dummy data (e.g., all 1.0s or random)
        // For simplicity, let's just use sequential values normalized
        for (size_t i = 0; i < input_tensor_size; i++) {
            input_tensor_values[i] = (float)i / (float)(input_tensor_size);
        }

        // Create Memory Info
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create Input Tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_tensor_values.data(), 
            input_tensor_size, 
            input_shape.data(), 
            input_shape.size()
        );

        // 5. Run Inference
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        std::cout << "Running inference..." << std::endl;
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, 
            output_names, 
            1
        );

        // 6. Process Output
        // Output should be [1, 10] (num_classes=10)
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::cout << "Inference successful!" << std::endl;
        std::cout << "Output elements (" << output_count << "):" << std::endl;
        
        for (size_t i = 0; i < output_count; i++) {
            std::cout << "Class " << i << ": " << floatarr[i] << std::endl;
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}