#ifndef __HISTOGRAM_GPU__
#define __HISTOGRAM_GPU__

#include "utils/image.hpp"
#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define HISTO_SIZE 101

class HistogramGPU {
	
	private:

		// CPU side
		Image _image;
		int _outRepart[HISTO_SIZE];
		float * _equal;
		int _imageSize;

		// GPU side
		unsigned char * _devInPixels = nullptr;
		unsigned char * _devOutPixels = nullptr;
		float * _devOutHue = nullptr;
		float * _devOutSaturation = nullptr;
		int * _devOutValue = nullptr;
		int * _devOutHisto = nullptr;
		int * _devOutRepart = nullptr;
		float * _devOutEqualisation = nullptr;

	public:

		HistogramGPU(const std::string & p_loadPath);
		~HistogramGPU();

		float histogramEqualisation(const std::string & p_loadPath, const std::string & p_SavePath, int* outGPU);

};

#endif // !__HISTOGRAM_GPU__