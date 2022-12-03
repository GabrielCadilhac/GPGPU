#include "histogramGPU.hpp"
#define HISTO_SIZE 101

__constant__ int repartition[HISTO_SIZE];

__global__ void rgb2hsv(unsigned char* p_devInPixels, int p_imageSize, float* p_outHue, float* p_outSaturation, int* p_outValue)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < p_imageSize)
	{
        float inRed   = static_cast<float>(p_devInPixels[3 * tid]) / 255.f;
        float inGreen = static_cast<float>(p_devInPixels[3 * tid + 1]) / 255.f;
        float inBlue  = static_cast<float>(p_devInPixels[3 * tid + 2]) / 255.f;

		float Cmax = fmax(inRed, fmax(inGreen, inBlue));
        float Cmin = fmin(inRed, fmin(inGreen, inBlue));
        float delta = Cmax - Cmin;
	
        if (delta == 0.0f)
            p_outHue[tid] = 0.f;
        else if (Cmax == inRed)
            p_outHue[tid] = fmod(60.f * ((inGreen - inBlue) / delta) + 360.f, 360.f);
        else if (Cmax == inGreen)
            p_outHue[tid] = 60.f * ((inBlue - inRed) / delta) + 120.f;
        else if (Cmax == inBlue)
            p_outHue[tid] = 60.f * ((inRed - inGreen) / delta) + 240.f;

        if (delta == 0)
            p_outSaturation[tid] = 0;
        else
            p_outSaturation[tid] = delta / Cmax;

        p_outValue[tid] = Cmax * 100;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void hsv2rgb(unsigned char* p_devOutPixels, int p_imageSize, float* p_inHue, float* p_inSaturation, int* p_inValue )
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < p_imageSize)
	{
		float v = static_cast<float>(p_inValue[tid])/100.f;
        float C = p_inSaturation[tid] * v;
        float X = C * (1.f - abs(fmod(p_inHue[tid] / 60.f, 2.f) - 1.f));
        float m = v - C;

        float r = 0.f;
        float g = 0.f;
        float b = 0.f;

        if (0 < p_inHue[tid] && p_inHue[tid] < 60)
            r = C, g = X, b = 0;
        else if (60 < p_inHue[tid] && p_inHue[tid] < 120)
            r = X, g = C, b = 0;
        else if (120 < p_inHue[tid] && p_inHue[tid] < 180)
            r = 0.f, g = C, b = X;
        else if (180 < p_inHue[tid] && p_inHue[tid] < 240)
            r = 0, g = X, b = C;
        else if (240 < p_inHue[tid] && p_inHue[tid] < 300)
            r = X, g = 0, b = C;
        else if (300 < p_inHue[tid] && p_inHue[tid] < 360)
            r = C, g = 0, b = X;

        p_devOutPixels[3 * tid]     = static_cast<unsigned char>(255 * (r + m));
        p_devOutPixels[3 * tid + 1] = static_cast<unsigned char>(255 * (g + m));
        p_devOutPixels[3 * tid + 2] = static_cast<unsigned char>(255 * (b + m));
		tid += blockDim.x * gridDim.x;
    }
}

__global__ void histogram(const int* const p_inValue, int *p_outHisto, const int p_valueSize)
{
	int tid = threadIdx.x;
	int N = 1;

    __shared__ int sharedHisto[HISTO_SIZE];

    while (tid < HISTO_SIZE)
    {
        sharedHisto[tid] = 0;
        tid += blockDim.x;
    }

    __syncthreads();
    tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < p_valueSize)
    {
        for (int i = tid*N; i < N*(tid+1) && i < p_valueSize; i++)
        {
            atomicAdd(sharedHisto+p_inValue[i], 1);
        }
        tid += gridDim.x * blockDim.x;
    }

    __syncthreads();
    tid = threadIdx.x;

    while (tid < HISTO_SIZE)
    {
        atomicAdd(p_outHisto+tid, sharedHisto[tid]);
        tid += blockDim.x;
    }
}

__global__ void repart(int* p_inHisto, int* p_inValue, int* p_outRepart)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int offset = 0;

	while (offset < HISTO_SIZE && tid >= offset && tid < HISTO_SIZE)
	{
		p_outRepart[tid] += p_inHisto[tid-offset];
		offset++;
	}
}

__global__ void equalization(const int* const p_inValue, int* const p_outEqualization, const int p_imageSize)
{
	const float LLn = 99.f / (100.f * p_imageSize);

	int tid = (threadIdx.x + blockIdx.x * blockDim.x);

	while (tid < p_imageSize);
	{
		int v = p_inValue[tid];
        p_outEqualization[tid] = (LLn * repartition[v]);
		tid += blockDim.x * gridDim.x;
    }
}

float HistogramGPU::histogramEqualisation(const std::string p_loadPath, const std::string p_savePath, int* outGPU)
{
	Image* image = new Image();
	image->load(p_loadPath);

	const int imageSize = image->_width * image->_height;

	unsigned char* devInPixels;
	unsigned char* devOutPixels;

	float* devOutHue;
	float* devOutSaturation;
	int* devOutValue;
	int* devOutHisto;
	int* devOutRepart;
	int* devOutEqualisation;

	int* outRepart = new int[HISTO_SIZE];
	int* outPixels = outGPU;

	HANDLE_ERROR(cudaMalloc((void**)&devInPixels, 3 * imageSize * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutPixels, 3 * imageSize * sizeof(unsigned char)));

	HANDLE_ERROR(cudaMalloc((void**)&devOutHue, imageSize * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutSaturation, imageSize * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutValue, imageSize * sizeof(int)));

	HANDLE_ERROR(cudaMalloc((void**)&devOutHisto, HISTO_SIZE * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutRepart, HISTO_SIZE * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutEqualisation, imageSize * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(devInPixels, image->_pixels, 3 * imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

	dim3 dimBlock(512);
	dim3 dimGrid((imageSize + dimBlock.x - 1) / dimBlock.x);

	std::cout << dimGrid.x * dimBlock.x << std::endl;
	ChronoGPU chr;
	chr.start();

	rgb2hsv <<<dimGrid,dimBlock>>>(devInPixels, imageSize, devOutHue, devOutSaturation, devOutValue);

	histogram<<<dimGrid,dimBlock>>>(devOutValue, devOutHisto, imageSize);
	
	repart<<<dimGrid,dimBlock>>>(devOutHisto,devOutValue,devOutRepart);

	HANDLE_ERROR(cudaMemcpy(outRepart, devOutRepart, HISTO_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpyToSymbol(repartition, devOutRepart,  HISTO_SIZE * sizeof(int)));

	equalization<<<dimGrid,dimBlock>>>(devOutValue, devOutEqualisation, imageSize);

	//hsv2rgb<<<dimGrid,dimBlock>>>(devOutPixels, imageSize, devOutHue, devOutSaturation, devOutEqualisation);
	
	chr.stop();
	
	HANDLE_ERROR(cudaMemcpy(outPixels, devOutEqualisation, imageSize * sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(devInPixels));
	HANDLE_ERROR(cudaFree(devOutPixels));
	HANDLE_ERROR(cudaFree(devOutHisto));
	HANDLE_ERROR(cudaFree(devOutEqualisation));
	HANDLE_ERROR(cudaFree(devOutRepart));
	HANDLE_ERROR(cudaFree(devOutHue));
	HANDLE_ERROR(cudaFree(devOutSaturation));
	HANDLE_ERROR(cudaFree(devOutValue));

	image->save(p_savePath);

	return chr.elapsedTime();
}
