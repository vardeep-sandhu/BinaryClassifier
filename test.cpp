#include <opencv2/opencv.hpp>
// #include <NvInferRuntime.h>
using namespace cv;

void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims)
{

    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);
    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    
    std::vector< cv::cuda::GpuMat > chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size)
{
    // get class names
    auto classes = getClassNames("imagenet_classes.txt");
 
    // copy results from GPU to CPU
    std::vector< float > cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector< int > indices(getSizeByDim(dims) * batch_size);
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "n";
        ++i;
    }
}

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "n";
        }
    }
} gLogger;
// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
 
template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr< nvinfer1::IExecutionContext >& context)
{
    TRTUniquePtr< nvinfer1::IBuilder > builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetwork()};
    TRTUniquePtr< nvonnxparser::IParser > parser{nvonnxparser::createParser(*network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast< int >(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    TRTUniquePtr< nvinfer1::IBuilderConfig > config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);

    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}