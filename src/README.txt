COMPILATION:

nvcc main.cu histogramGPU.CU histogramCPU.cpp utils/chronoCPU.cpp utils/chronoGPU.cu utils/image.cpp -O3 -o main 

EXEMPLE EXECUTION:

// -N 4 valeurs par 4, -b 128 threads par block, -i url de l'image -> images/batiment-1.jpg
main.exe -N 4 -b 128 -i ./images/batiment-1.jpg