#include <iostream>
#include <string>
#include "test.cpp"

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx image.jpgn";
        return -1;
    }
    std::string model_path = "/home/sandhu/project/BinaryClassifier/resnet50.onnx";
    std::string image_path = "/home/sandhu/project/BinaryClassifier/dataset/valid/0_y0310_x0914.png";
    int batch_size = 1;
    TRTUniquePtr< nvinfer1::ICudaEngine > engine{nullptr};
    TRTUniquePtr< nvinfer1::IExecutionContext > context{nullptr};
    parseOnnxModel(model_path, engine, context);

    std::vector< nvinfer1::Dims > input_dims; // we expect only one input
    std::vector< nvinfer1::Dims > output_dims; // and one output
    std::vector< void* > buffers(engine->getNbBindings()); // buffers for input and output data
for (size_t i = 0; i < engine->getNbBindings(); ++i)
{
    auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
    cudaMalloc(&buffers[i], binding_size);
    if (engine->bindingIsInput(i))
    {
        input_dims.emplace_back(engine->getBindingDimensions(i));
    }
    else
    {
        output_dims.emplace_back(engine->getBindingDimensions(i));
    }
}
if (input_dims.empty() || output_dims.empty())
{
    std::cerr << "Expect at least one input and one output for networkn";
    return -1;
}

// preprocess input data
    PreprocessImage(image_path, (float*)buffers[0], input_dims[0]);
    // inference
    context->enqueue(batch_size, buffers.data(), 0, nullptr);
    // post-process results
    PostprocessResults((float *) buffers[1], output_dims[0], batch_size);
 
    for (void* buf : buffers)
    {
        cudaFree(buf);
    }
    return 0;
}