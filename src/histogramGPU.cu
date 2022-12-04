#include "histogramGPU.hpp"

// CONSTANTS GPU
__constant__ int GPU_REPARTITION[HISTO_SIZE];

HistogramGPU::HistogramGPU(const std::string & p_imageLoadPath)
{
	_image.load(p_imageLoadPath);
	_imageSize = _image._width * _image._height;
	_equal = new float[_imageSize];

	HANDLE_ERROR(cudaMalloc((void**)&_devInPixels, 3 * _imageSize * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&_devOutPixels, 3 * _imageSize * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&_devOutHue, _imageSize * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&_devOutSaturation, _imageSize * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&_devOutValue, _imageSize * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&_devOutHisto, HISTO_SIZE * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&_devOutRepart, HISTO_SIZE * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&_devOutEqualisation, _imageSize * sizeof(float)));
}

HistogramGPU::~HistogramGPU()
{
	delete[] _equal;

	HANDLE_ERROR(cudaFree(_devInPixels));
	HANDLE_ERROR(cudaFree(_devOutPixels));
	HANDLE_ERROR(cudaFree(_devOutHisto));
	HANDLE_ERROR(cudaFree(_devOutEqualisation));
	HANDLE_ERROR(cudaFree(_devOutRepart));
	HANDLE_ERROR(cudaFree(_devOutHue));
	HANDLE_ERROR(cudaFree(_devOutSaturation));
	HANDLE_ERROR(cudaFree(_devOutValue));
}

__global__ void rgb2hsv(unsigned char* p_devInPixels, int p_imageSize, float* p_outHue, float* p_outSaturation, int* p_outValue)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(int id = tid; id < p_imageSize; id += blockDim.x * gridDim.x)
	{
        float inRed   = static_cast<float>(p_devInPixels[3 * id]) / 255.f;
        float inGreen = static_cast<float>(p_devInPixels[3 * id + 1]) / 255.f;
        float inBlue  = static_cast<float>(p_devInPixels[3 * id + 2]) / 255.f;

		float Cmax = fmax(inRed, fmax(inGreen, inBlue));
        float Cmin = fmin(inRed, fmin(inGreen, inBlue));
        float delta = Cmax - Cmin;
	
        if (delta == 0.0f)
            p_outHue[id] = 0.f;
        else if (Cmax == inRed)
            p_outHue[id] = 60.f * ((inGreen - inBlue) / delta);
        else if (Cmax == inGreen)
            p_outHue[id] = 60.f * (((inBlue - inRed) / delta) + 2.f);
        else if (Cmax == inBlue)
            p_outHue[id] = 60.f * (((inRed - inGreen) / delta) + 4.f);

		if (p_outHue[id] < 0)
			p_outHue[id] += 360.f;

        if (Cmax > 0)
			p_outSaturation[id] = delta / Cmax;
        else
			p_outSaturation[id] = 0.0f;

        p_outValue[id] = Cmax * 100;
	}
}

__global__ void hsv2rgb(unsigned char* p_devOutPixels, int p_imageSize, float* p_inHue, float* p_inSaturation, float* p_inValue )
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	for(int id = tid; id < p_imageSize; id += blockDim.x * gridDim.x)
	{
		float v = p_inValue[id];
        float C = p_inSaturation[id] * v;
        float X = C * (1.f - abs(fmod(p_inHue[id] / 60.f, 2.f) - 1.f));
        float m = v - C;

        float r = 0.f;
        float g = 0.f;
        float b = 0.f;

        if (p_inHue[id] < 60)
            r = C, g = X, b = 0;
        else if (p_inHue[id] < 120)
            r = X, g = C, b = 0;
        else if (p_inHue[id] < 180)
            r = 0.f, g = C, b = X;
        else if (p_inHue[id] < 240)
            r = 0, g = X, b = C;
        else if (p_inHue[id] < 300)
            r = X, g = 0, b = C;
		else
            r = C, g = 0, b = X;

        p_devOutPixels[3 * id]     = static_cast<unsigned char>(255.0f * (r + m));
        p_devOutPixels[3 * id + 1] = static_cast<unsigned char>(255.0f * (g + m));
        p_devOutPixels[3 * id + 2] = static_cast<unsigned char>(255.0f * (b + m));
    }
}

__global__ void histogram(int * p_inValue, int * p_outHisto, int p_valueSize)
{
	__shared__ int sharedHisto[HISTO_SIZE];
	int tid = threadIdx.x;
	int N = 1;

    for(int id = tid; id < HISTO_SIZE; id += blockDim.x)
    {
        sharedHisto[id] = 0;
    }
    __syncthreads();

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int id = tid; id < p_valueSize; id += gridDim.x * blockDim.x)
    {
        for (int i = id*N; i < N*(id+1) && i < p_valueSize; i++)
        {
            atomicAdd(sharedHisto + p_inValue[i], 1);
        }
    }
    __syncthreads();

    tid = threadIdx.x;
    for(int id = tid; id < HISTO_SIZE; id += blockDim.x)
    {
        atomicAdd(p_outHisto + id, sharedHisto[id]);
    }
}

__global__ void repart(int * p_inHisto, int * p_inValue, int * p_outRepart)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int offset = 0;

	while (offset < HISTO_SIZE && tid >= offset && tid < HISTO_SIZE)
	{
		p_outRepart[tid] += p_inHisto[tid-offset];
		offset++;
	}
}

__global__ void equalization(int * p_inValue, float * p_outEqualization, const int p_imageSize)
{
	const float LLn = 99.f / (100.f * p_imageSize);
	int tid = (threadIdx.x + blockIdx.x * blockDim.x);

	for(int id = tid; id < p_imageSize; id += blockDim.x * gridDim.x)
	{
		int v = p_inValue[id];
        p_outEqualization[id] = (LLn * GPU_REPARTITION[v]);
    }
}

float HistogramGPU::histogramEqualisation(const std::string & p_loadPath, const std::string & p_savePath, int * result)
{
	HANDLE_ERROR(cudaMemcpy(_devInPixels, _image._pixels, 3 * _imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

	// Configure amount of threads and blocks
	dim3 dimBlock(512);
	dim3 dimGrid((_imageSize + dimBlock.x - 1) / dimBlock.x);

	// Start equalization
	ChronoGPU chr;
	chr.start();
	rgb2hsv <<<dimGrid,dimBlock>>>(_devInPixels, _imageSize, _devOutHue, _devOutSaturation, _devOutValue);
	histogram<<<dimGrid,dimBlock>>>(_devOutValue, _devOutHisto, _imageSize);
	repart<<<dimGrid,dimBlock>>>(_devOutHisto, _devOutValue, _devOutRepart);
	HANDLE_ERROR(cudaMemcpy(_outRepart, _devOutRepart, HISTO_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpyToSymbol(GPU_REPARTITION, _devOutRepart,  HISTO_SIZE * sizeof(int)));
	equalization<<<dimGrid,dimBlock>>>(_devOutValue, _devOutEqualisation, _imageSize);
	hsv2rgb<<<dimGrid,dimBlock>>>(_devOutPixels, _imageSize, _devOutHue, _devOutSaturation, _devOutEqualisation);
	chr.stop();
	
	// Send the GPU result to the CPU memory
	HANDLE_ERROR(cudaMemcpy(_image._pixels, _devOutPixels, 3 * _imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(_equal, _devOutEqualisation, _imageSize * sizeof(int), cudaMemcpyDeviceToHost));

	// Map [0, 1] float to (0, 255) RGB values
	for(unsigned int i = 0; i < _imageSize; i++)
	{
		result[i] = static_cast<int>(_equal[i] * 255);
	}
	
	// Save the image
	_image.save(p_savePath);

	return chr.elapsedTime();
}
