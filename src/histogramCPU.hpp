#ifndef __HISTOGRAM_CPU__
#define __HISTOGRAM_CPU__

#include "utils/image.hpp"
#include "utils/chronoCPU.hpp"
#include <math.h>
#include <iostream>

class HistogramCPU {
	public :
		HistogramCPU() = default;

		int _imageSize = 0;
		int _nbChannels = 3;

		float* _hue = nullptr;
		float* _saturation = nullptr;
		int* _value = nullptr;
	
		unsigned char* _pixels = nullptr;

		void rgb2hsv(unsigned char* p_pixels);
		void hsv2rgb(unsigned char* p_rgb);
	
		int* histogram();
		float repart(int* p_hist, int p_l);
		float* equalization();

		float histogramEqualisation(const std::string & p_imageLoadPath, const std::string & p_imageSavePath, unsigned char* outCPU);
};

#endif // ! __HISTOGRAM_CPU__


