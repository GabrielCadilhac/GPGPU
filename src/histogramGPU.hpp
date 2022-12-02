#ifndef __HISTOGRAM_GPU__
#define __HISTOGRAM_GPU__

#include "utils/image.hpp"
#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"

#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

class HistogramGPU {
	public:
		float histogramEqualisation(const std::string p_loadPath, const std::string p_SavePath, int* outGPU);
};

#endif // !__HISTOGRAM_GPU__