#include "histogramCPU.hpp"

HistogramCPU::HistogramCPU(const std::string & p_imageLoadPath)
{
    _image.load(p_imageLoadPath);
    _imageSize = _image._width * _image._height;
    _nbChannels = _image._nbChannels;

    _histo = new int[histoSize + 1];
    _repart = new int[histoSize + 1];
    _hue = new float[_imageSize];
    _saturation = new float[_imageSize];
    _value = new int[_imageSize];
}

HistogramCPU::~HistogramCPU()
{
    delete[] _histo;
    delete[] _hue;
    delete[] _saturation;
    delete[] _value;
    delete[] _equal;
}

void HistogramCPU::rgb2hsv(unsigned char * p_pixel)
{
    ChronoCPU chronoCPU;
    chronoCPU.start();

    float rgb[3];
    for (unsigned int i = 0; i < _imageSize; i++)
    {
        rgb[0] = static_cast<float>(p_pixel[3 * i]) / 255.f;
        rgb[1] = static_cast<float>(p_pixel[3 * i + 1]) / 255.f;
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

        while (_hue[i] < 0)
            _hue[i] += 360.f;

        _saturation[i] = delta / Cmax;
        _value[i] = Cmax*100.f;
    }

    chronoCPU.stop();
    std::cout << "HistogramCPU::rgb2hsv -> " << chronoCPU.elapsedTime() << " ms" << std::endl;
}

void HistogramCPU::hsv2rgb(unsigned char* p_rgb)
{
    ChronoCPU chronoCPU;
    chronoCPU.start();

    for (unsigned int i = 0; i < _imageSize; i++)
    {
        float v = _equal[i];
        float C = _saturation[i] * v;
        float X = C * (1.f - std::abs(std::fmod(_hue[i] / 60.f, 2.f) - 1.f));
        float m = v - C;

        float r = C;
        float g = 0.f;
        float b = X;

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

        p_rgb[3*i] = static_cast<unsigned char>(255 * (r + m));
        p_rgb[3*i+1] = static_cast<unsigned char>(255 * (g + m));
        p_rgb[3*i+2] = static_cast<unsigned char>(255 * (b + m));
    }

    chronoCPU.stop();
    std::cout << "HistogramCPU::hsv2rgb -> " << chronoCPU.elapsedTime() << " ms" << std::endl;
}

void HistogramCPU::histogram()
{
    ChronoCPU chronoCPU;
    chronoCPU.start();

    for (unsigned int i = 0; i <= histoSize; i++)
        _histo[i] = 0;

    for (unsigned int i = 0; i < _imageSize; i++)
    {
        int V = _value[i];
        _histo[V]++;
    }

    chronoCPU.stop();
    std::cout << "HistogramCPU::histogram -> " << chronoCPU.elapsedTime() << " ms" << std::endl;
}

void HistogramCPU::repart()
{
    ChronoCPU chronoCPU;
    chronoCPU.start();

    _repart[0] = _histo[0]; 

    for (unsigned int i = 1; i < 101; i++)
    {
        _repart[i] = _repart[i-1] + _histo[i];    
    }

    chronoCPU.stop();
    std::cout << "HistogramCPU::repart -> " << chronoCPU.elapsedTime() << " ms" << std::endl;
}

void HistogramCPU::equalization()
{
    ChronoCPU chronoCPU;
    chronoCPU.start();

    histogram();
    int sum = 0;
    for(unsigned int i = 0; i < 101; i++)
    {
        sum += _histo[i];
    }
    repart();

    _equal = new float[_imageSize];
    float LLn = 99.f / (100.f * _imageSize);

    for (unsigned int i = 0; i < _imageSize; i++)
    {
        int v = _value[i];
        _equal[i] = (LLn * _repart[v]);
    }

    chronoCPU.stop();
    std::cout << "HistogramCPU::equalization -> " << chronoCPU.elapsedTime() << " ms" << std::endl;
}

float HistogramCPU::histogramEqualisation(const std::string & p_imageSavePath, int * result)
{
    // Start equalization
    ChronoCPU chronoCPU;
    chronoCPU.start();
    rgb2hsv(_image._pixels);
    equalization();
    hsv2rgb(_image._pixels);
    chronoCPU.stop();
    
    // Map [0, 1] values to [0, 255]
    for (unsigned int i = 0; i < _imageSize; ++i)
    {
        result[i] = (int) (255 * _equal[i]);
    }

    // Save image
    _image.save(p_imageSavePath);

    return chronoCPU.elapsedTime();
}