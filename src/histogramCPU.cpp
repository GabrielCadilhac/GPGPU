#include "histogramCPU.hpp"


void HistogramCPU::rgb2hsv(unsigned char* p_pixel)
{
    for (int i = 0; i < _imageSize; i++)
    {
        float* rgb = new float[3];
        rgb[0] = static_cast<float>(p_pixel[3 * i])/255.f;
        rgb[1] = static_cast<float>(p_pixel[3 * i + 1])/255.f;
        rgb[2] = static_cast<float>(p_pixel[3 * i + 2]) / 255.f;

        float Cmax = std::fmax(rgb[0], std::fmax(rgb[1], rgb[2]));
        float Cmin = std::fmin(rgb[0], std::fmin(rgb[1], rgb[2]));
        float delta = Cmax - Cmin;

        if (delta == 0.f)
            _hue[i] = 0.f;
        else if (Cmax == rgb[0])
            _hue[i] = 60.f * ((rgb[1] - rgb[2]) / delta);
        else if (Cmax == rgb[1])
            _hue[i] = 60.f * (((rgb[2] - rgb[0]) / delta) + 2.f);
        else if (Cmax == rgb[2])
            _hue[i] = 60.f * (((rgb[0] - rgb[1]) / delta) + 4.f);

        if (_hue[i] < 0)
            _hue[i] += 360.f;

        if (Cmax > 0)
            _saturation[i] = delta / Cmax;
        else
            _saturation[i] = 0.f;

        _value[i] = Cmax*100.f;

        delete[] rgb;
    }
}

void HistogramCPU::hsv2rgb(unsigned char* p_rgb)
{
    std::cout << _equal[0] << std::endl;  
    for (int i = 0; i < _imageSize; i++)
    {
        float v = _equal[i];
        float C = _saturation[i] * v;
        float X = C * (1.f - std::abs(std::fmod(_hue[i] / 60.f, 2.f) - 1.f));
        float m = v - C;

        float r = 0.f;
        float g = 0.f;
        float b = 0.f;

        if (_hue[i] < 60)
            r = C, g = X, b = 0;
        else if (_hue[i] < 120)
            r = X, g = C, b = 0;
        else if (_hue[i] < 180)
            r = 0, g = C, b = X;
        else if (_hue[i] < 240)
            r = 0, g = X, b = C;
        else if (_hue[i] < 300)
            r = X, g = 0, b = C;
        else
            r = C, g = 0, b = X;

        p_rgb[3*i] = static_cast<unsigned char>(255 * (r + m));
        p_rgb[3*i+1] = static_cast<unsigned char>(255 * (g + m));
        p_rgb[3*i+2] = static_cast<unsigned char>(255 * (b + m));
    }
}

void HistogramCPU::histogram()
{
    _histo = new int[histoSize+1];

    for (int i = 0; i <= histoSize; i++)
        _histo[i] = 0;

    for (int i = 0; i < _imageSize; i++)
    {
        int V = _value[i];
        _histo[V]++;
    }
}

void HistogramCPU::repart()
{
    _repart = new int[101];
    
    _repart[0] = _histo[0]; 

    for (int i = 1; i < 101; i++)
    {
        _repart[i] = _repart[i-1] + _histo[i];    
    }

    std::cout << "repart" << _repart[histoSize] << std::endl; 
}

void HistogramCPU::equalization()
{
    histogram();

    int sum = 0;
    for(int i = 0; i < 101; i++)
        sum += _histo[i];

    repart();

    _equal = new float[_imageSize];
    float LLn = 99.f / (100.f * _imageSize);

    for (int i = 0; i < _imageSize; i++)
    {
        int v = _value[i];
        _equal[i] = (LLn * _repart[v]);
    }
}

float HistogramCPU::histogramEqualisation(const std::string &p_imageLoadPath, const std::string& p_imageSavePath, int* outCPU)
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
    
    equalization();
    
    hsv2rgb(image->_pixels);
    
    chronoCPU.stop();

    for (int i = 0; i < _imageSize; ++i)
    {
        outCPU[i] = _equal[i];
    }

    image->save(p_imageSavePath);
    return chronoCPU.elapsedTime();
}