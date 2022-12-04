#ifndef __HISTOGRAM_CPU__
#define __HISTOGRAM_CPU__

#include "utils/image.hpp"
#include "utils/chronoCPU.hpp"
#include <math.h>
#include <iostream>

class HistogramCPU {

	private:

		// Constants
		const int histoSize = 100;

		// Image attributes
		Image _image;
		int _imageSize = 0;
		int _nbChannels = 3;
		unsigned char* _pixels = nullptr;

		// HSV pixels 
		float* _hue = nullptr;
		float* _saturation = nullptr;
		int* _value = nullptr;

		// Histogram
		int* _histo = nullptr;
		int* _repart = nullptr;
		float* _equal = nullptr;

	public :
		HistogramCPU(const std::string & p_imageLoadPath);
		~HistogramCPU();

		// Methods
		void rgb2hsv(unsigned char * p_pixels);
		void hsv2rgb(unsigned char * p_rgb);
		void histogram();
		void repart();
		void equalization();
		float histogramEqualisation(const std::string & p_imageSavePath, int* outCPU);
};

#endif // ! __HISTOGRAM_CPU__


