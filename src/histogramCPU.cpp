#include "histogramCPU.hpp"

void HistogramCPU::rgb2hsv(unsigned char* p_pixel)
{
    for (int i = 0; i < _imageSize; i++)
    {
        float* rgb = new float[3];
        rgb[0] = static_cast<float>(p_pixel[3 * i]) / 255.f;
        rgb[1] = static_cast<float>(p_pixel[3 * i + 1]) / 255.f;
        rgb[2] = static_cast<float>(p_pixel[3 * i + 2]) / 255.f;
        
        float Cmax = std::fmax(rgb[0], std::fmax(rgb[1], rgb[2]));
        float Cmin = std::fmin(rgb[0], std::fmin(rgb[1], rgb[2]));
        float delta = Cmax - Cmin;

        if (delta == 0.0f)
            _hue[i] = 0.f;
        else if (Cmax == rgb[0])
            _hue[i] = static_cast<int>(60.f * ((rgb[1] - rgb[2]) / delta) + 360.f) % 360;
        else if (Cmax == rgb[1])
            _hue[i] = 60.f * ((rgb[2] - rgb[0]) / delta) + 120.f;
        else if (Cmax == rgb[2])
            _hue[i] = 60.f * ((rgb[0] - rgb[1]) / delta) + 240.f;

        if (delta == 0)
            _saturation[i] = 0;
        else
            _saturation[i] = delta / Cmax;

        _value[i] = Cmax*100;
    }
}

void HistogramCPU::hsv2rgb(unsigned char* p_rgb)
{
    for (int i = 0; i < _imageSize; i++)
    {
        float C = _saturation[i] * _value[i];
        float X = C * (1.f - std::abs(std::fmodf(_hue[i] / 60, 2.f) - 1.f));
        float m = _value[i] - C;

        float r = 0.f;
        float g = 0.f;
        float b = 0.f;

        if (0 < _hue[i] && _hue[i] < 60)
            r = C, g = X, b = 0;
        else if (60 < _hue[i] && _hue[i] < 120)
            r = X, g = C, b = 0;
        else if (120 < _hue[i] && _hue[i] < 180)
            r = 0, g = C, b = X;
        else if (180 < _hue[i] && _hue[i] < 240)
            r = 0, g = X, b = C;
        else if (240 < _hue[i] && _hue[i] < 300)
            r = X, g = 0, b = C;
        else if (300 < _hue[i] && _hue[i] < 360)
            r = C, g = 0, b = X;

        p_rgb[3 * i] = static_cast<unsigned char>(255 * (r + m));
        p_rgb[3 * i + 1] = static_cast<unsigned char>(255 * (g + m));
        p_rgb[3 * i + 2] = static_cast<unsigned char>(255 * (b + m));
    }
}

int* HistogramCPU::histogram()
{
    int* hist = new int[101];

    for (int i = 0; i <= 100; i++)
        hist[i] = 0;

    for (int i = 0; i < _imageSize; i++)
    {
        int V = _value[i] * 100;
        hist[V]++;
    }

    return hist;
}

float HistogramCPU::repart(int* p_hist, int p_l)
{
    float r = 0.f;
    for (int k = 0; k <= p_l; k++)
        r += p_hist[k];
    
    return r;
}

float* HistogramCPU::equalization()
{
    int* hist = histogram();
    float* T = new float[_imageSize];
    float LLn = 99.f / (100.f * _imageSize);

    for (int i = 0; i < _imageSize; i++)
    {
        T[i] = (LLn * repart(hist, _value[i] * 100));
    }

    return T;
}

float HistogramCPU::histogramEqualisation(const std::string &p_imageLoadPath, const std::string& p_imageSavePath, unsigned char* outCPU)
{
    Image* image = new Image();
    image->load(p_imageLoadPath);
    _imageSize = image->_width * image->_height;
    _nbChannels = image->_nbChannels;

    _hue = new float[_imageSize];
    _saturation = new float[_imageSize];
    _value = new int[_imageSize];

    ChronoCPU chronoCPU;
    chronoCPU.start();
    rgb2hsv(image->_pixels);
    chronoCPU.stop();
    /*
    float* newValues = equalization();
    */
    hsv2rgb(image->_pixels);
    
    for (int i = 0; i < _imageSize*3; ++i)
    {
        outCPU[i] = image->_pixels[i];
    }

    image->save(p_imageSavePath);
    return chronoCPU.elapsedTime();
}