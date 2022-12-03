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

		int histoSize = 100;
		int* _histo = nullptr;
		int* _repart = nullptr;
		float* _equal = nullptr;

		void rgb2hsv(unsigned char* p_pixels);
		void hsv2rgb(unsigned char* p_rgb);
	
		void histogram();
		void repart();
		void equalization();

		float histogramEqualisation(const std::string & p_imageLoadPath, const std::string & p_imageSavePath, int* outCPU);
};

#endif // ! __HISTOGRAM_CPU__


