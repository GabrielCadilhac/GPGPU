#include "utils/image.hpp"
#include <math.h>
#include <iostream>
#include "histogramCPU.hpp"
#include "histogramGPU.hpp"

#define OUT_CPU_IMAGE_PATH "./images/gpu-result.jpg"
#define OUT_GPU_IMAGE_PATH "./images/cpu-result.jpg"

int main(int argc, char const *argv[])
{
    int N = 4;
    int blockSize = 512;
    std::string path("./images/batiment-1.jpg");

    if(argc <= 1)
    {
        std::cout << "Missing argument !" << std::endl;
    }
    for (int i = 1; i < argc; ++i)
    {
        if ( !strcmp(argv[i], "-N"))
        {
            if (sscanf(argv[++i], "%i", &N) != 1)
                std::cout << "Error argument N" << std::endl;
        }
        else if ( !strcmp(argv[i], "-b"))
        {
            if (sscanf(argv[++i], "%i", &blockSize) != 1)
                std::cout << "Error argument b" << std::endl;
        }
        else if ( !strcmp(argv[i], "-i"))
        {
            path = std::string(argv[++i]);
            std::cout << "Image loaded -> " << path << std::endl;
        }
    }

    // Get the image width and height
    Image image;
    image.load(path.c_str());

    // Results
    const unsigned int imageSize = image._width * image._height;
    int * outCPU = new int[imageSize];
    int * outGPU = new int[imageSize];

    // CPU sequential
	std::cout << "============================================"	<< std::endl;
	std::cout << "         Sequential version on CPU          "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

    float cpuTime = 0.f;
    HistogramCPU histogramCPU(path.c_str());
    cpuTime = histogramCPU.histogramEqualisation(OUT_CPU_IMAGE_PATH, outCPU);
    std::cout << "Time : " << cpuTime << std::endl;
    
    // GPU CUDA
	std::cout << "============================================"	<< std::endl;
	std::cout << "         Parallel version on GPU            "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

    float gpuTime = 0.f;
    HistogramGPU histogramGPU(path.c_str());
    gpuTime = histogramGPU.histogramEqualisation(OUT_GPU_IMAGE_PATH, outGPU, N, blockSize);
    std::cout << "Time : " << gpuTime << std::endl;

    // CHECK RESULT
    std::cout << "============================================"	<< std::endl;
	std::cout << "              Checking results		      "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

	for ( int i = 0; i < imageSize; ++i ) 
	{
        // Result may be slightly different between CPU and GPU because of the floating-point calculation
        if ( abs(outCPU[i] - outGPU[i]) > 2)  { 
            std::cerr << "Error for index " << i <<" :" << std::endl;
            std::cerr << "\t CPU: [" << outCPU[i] << "]" << std::endl;
            std::cerr << "\t GPU: [" << outGPU[i] << "]" << std::endl;
            exit( EXIT_FAILURE );
        }
	}
	std::cout << "Congratulations! Job's done!" << std::endl << std::endl;

    // FREE MEMORY
    delete[] outCPU;
    delete[] outGPU;

    return 0;
}
