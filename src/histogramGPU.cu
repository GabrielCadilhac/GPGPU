#include "histogramGPU.hpp"

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
	int bid = blockIdx.x + gridDim.x * blockIdx.y; // Index d'un block dans une grille 2D
	int totalBlock = gridDim.x * gridDim.y; // Nombre de block total
	int tid = (threadIdx.x + bid * blockDim.x); // Index du thread global (dans une grille 2D)
	
	while (tid < p_valueSize)
    {
        atomicAdd(&p_outHisto[p_inValue[tid]], 1);
		tid += blockDim.x * totalBlock;
    }
}

__global__ void repart(int* p_inHisto, int* p_outRepart, const int p_histoSize)
{
	extern __shared__ int sharedHisto[];

	int bid = blockIdx.x + gridDim.x * blockIdx.y; // Index d'un block dans une grille 2D
	int tid = threadIdx.x; // Index du thread global (dans une grille 2D)
	int tidGlobal = (threadIdx.x + bid * blockDim.x);

	sharedHisto[tid] = p_inHisto[tidGlobal];
	__syncthreads();

	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if (tid % (2*i) == 0)
			sharedHisto[tid] += sharedHisto[tid - i];
		__syncthreads();
	}

	if (tid == 0) 
		p_outRepart[blockIdx.x * blockIdx.y] = sharedHisto[0];
}

__global__ void equalization(const int* const p_inRepart, int* const p_outEqualization, const int p_imageSize)
{
	const float LLn = 99.f / (100.f * p_imageSize);

	int bid = blockIdx.x + gridDim.x * blockIdx.y; // Index d'un block dans une grille 2D
	int totalBlock = gridDim.x * gridDim.y; // Nombre de block total
	int tid = (threadIdx.x + bid * blockDim.x); // Index du thread global (dans une grille 2D)

	while (tid < p_imageSize);
	{
        p_outEqualization[tid] = (LLn * p_inRepart[tid]);
		tid += blockDim.x * totalBlock;
    }
}

float HistogramGPU::histogramEqualisation(const std::string p_loadPath, const std::string p_savePath, unsigned char* outGPU)
{
	Image* image = new Image();
	image->load(p_loadPath);

	const int imageSize = image->_width * image->_height;

	unsigned char* devInPixels;
	unsigned char* devOutPixels;

	float* devOutHue;
	float* devOutSaturation;
	int* devOutValue;

	float* outHue = new float[imageSize];
	float* outSaturation = new float[imageSize];
	int* outValue = new int[imageSize];

	unsigned char *outPixels = outGPU; 

	HANDLE_ERROR(cudaMalloc((void**)&devInPixels, 3 * imageSize * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutPixels, 3 * imageSize * sizeof(unsigned char)));

	HANDLE_ERROR(cudaMalloc((void**)&devOutHue, imageSize * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutSaturation, imageSize * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&devOutValue, imageSize * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(devInPixels, image->_pixels, 3 * imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

	dim3 dimBlock(1024);
	dim3 dimGrid((imageSize + dimBlock.x - 1) / dimBlock.x);

	ChronoGPU chr;
	chr.start();

	rgb2hsv <<<dimGrid,dimBlock>>>(devInPixels, imageSize, devOutHue, devOutSaturation, devOutValue);

	hsv2rgb<<<dimGrid,dimBlock>>>(devOutPixels, imageSize, devOutHue, devOutSaturation, devOutValue);
	chr.stop();

	HANDLE_ERROR(cudaMemcpy(outPixels, devOutPixels, 3 * imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(outHue, devOutHue, imageSize * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(outSaturation, devOutSaturation, imageSize * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(outValue, devOutValue, imageSize * sizeof(int), cudaMemcpyDeviceToHost));

	std::cout << outValue[0] << std::endl;

	HANDLE_ERROR(cudaFree(devInPixels));
	HANDLE_ERROR(cudaFree(devOutPixels));
	HANDLE_ERROR(cudaFree(devOutHue));
	HANDLE_ERROR(cudaFree(devOutSaturation));
	HANDLE_ERROR(cudaFree(devOutValue));

	image->save(p_savePath);

	return chr.elapsedTime();
}
