#include "utils/image.hpp"
#include <math.h>
#include <iostream>
#include "histogramCPU.hpp"
#include "histogramGPU.hpp"

int main(int argc, char const *argv[])
{
    Image *image = new Image();
    image->load("./images/chevre.png");
    const int imageSize = image->_width * image->_height;

    int *outCPU = new int[imageSize];
    int *outGPU = new int[imageSize];

    float cpuTime = 0.f;
    float gpuTime = 0.f;

    // CPU sequential
	std::cout << "============================================"	<< std::endl;
	std::cout << "         Sequential version on CPU          "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

    HistogramCPU histCPU;
    cpuTime = histCPU.histogramEqualisation("./images/chevre.png", "./images/chevreCPU.png", outCPU);
    std::cout << "Time : " << cpuTime << std::endl;
    
    // GPU CUDA
	/*std::cout << "============================================"	<< std::endl;
	std::cout << "         Parallel version on GPU            "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

    HistogramGPU histGPU;
    gpuTime = histGPU.histogramEqualisation("./images/chevre.png", "./images/chevreGPU.png", outGPU);
    std::cout << "Time : " << gpuTime << std::endl;

    std::cout << "============================================"	<< std::endl;
	std::cout << "              Checking results		      "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

	for ( int i = 0; i < imageSize; ++i ) 
	{
        // Result may be slightly different between CPU and GPU because of the floating-point calculation
        if ( fabsf(outCPU[i] - outGPU[i]) > 1)  { 
            std::cerr << "Error for index " << i <<" :" << std::endl;
            std::cerr << "\t CPU: [" << static_cast<int>(outCPU[i]) << "]" << std::endl;
            std::cerr << "\t GPU: [" << static_cast<int>(outGPU[i]) << "]" << std::endl;
            exit( EXIT_FAILURE );
        }
	}
	std::cout << "Congratulations! Job's done!" << std::endl << std::endl;*/

    delete[] outCPU;
    delete[] outGPU;
    return 0;
}
