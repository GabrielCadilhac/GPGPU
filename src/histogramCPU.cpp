#include "histogramCPU.hpp"
void rgb2hsv(unsigned char* p_pixels, int p_imageSize, float* p_hue, float* p_saturation, float* p_value)
{
    for (int i = 0; i < p_imageSize; i++)
    {
        float* rgb = new float[3];
        rgb[0] = static_cast<float>(p_pixels[3 * i])/255.f;
        rgb[1] = static_cast<float>(p_pixels[3 * i + 1])/255.f;
        rgb[2] = static_cast<float>(p_pixels[3 * i + 2]) / 255.f;

        float Cmax = std::max(rgb[0], std::max(rgb[1], rgb[2]));
        float Cmin = std::min(rgb[0], std::min(rgb[1], rgb[2]));
        float delta = Cmax - Cmin;

        if (delta == 0.f)
            p_hue[i] = 0.f;
        else if (Cmax == rgb[0])
            p_hue[i] = 60.f * ((rgb[1] - rgb[2]) / delta);
        else if (Cmax == rgb[1])
            p_hue[i] = 60.f * (((rgb[2] - rgb[0]) / delta) + 2.f);
        else if (Cmax == rgb[2])
            p_hue[i] = 60.f * (((rgb[0] - rgb[1]) / delta) + 4.f);

        if (p_hue[i] < 0)
            p_hue[i] += 360;

        if (Cmax > 0)
            p_saturation[i] = delta / Cmax;
        else
            p_saturation[i] = 0.f;

        p_value[i] = Cmax;

        delete[] rgb;
    }
}

void hsv2rgb(unsigned char* p_rgb, int p_imageSize, float* p_hue, float* p_saturation, float* p_value) {
    for (int i = 0; i < p_imageSize; i++)
    {
        float C = p_saturation[i] * p_value[i];
        float X = C * (1.f - std::abs(std::fmod(p_hue[i] / 60.f, 2.f) - 1.f));
        float m = p_value[i] - C;

        float r = 0.f;
        float g = 0.f;
        float b = 0.f;

        if (p_hue[i] < 60)
            r = C, g = X, b = 0;
        else if (p_hue[i] < 120)
            r = X, g = C, b = 0;
        else if (p_hue[i] < 180)
            r = 0, g = C, b = X;
        else if (p_hue[i] < 240)
            r = 0, g = X, b = C;
        else if (p_hue[i] < 300)
            r = X, g = 0, b = C;
        else
            r = C, g = 0, b = X;

        p_rgb[3*i] = static_cast<unsigned char>(255 * (r + m));
        p_rgb[3*i+1] = static_cast<unsigned char>(255 * (g + m));
        p_rgb[3*i+2] = static_cast<unsigned char>(255 * (b + m));
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