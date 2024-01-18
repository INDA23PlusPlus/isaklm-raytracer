# Path tracer written using CUDA and OpenGL

Supports:
- unbiased rendering
- dielectric, metallic and transparent materials
- OBJ model loading + custom material file format
- K-D tree acceleration structure
- importance sampling via Next Event Estimation
- adaptive sampling
- ACES tone mapping

## Example of a render (2 million triangles, 5000 samples per pixel)
![path_tracer_render](https://github.com/INDA23PlusPlus/isaklm-raytracer/assets/71440182/66864995-02e3-4bba-a15d-a24eb718eb05)

## How to run the project

To run the project:
- download Visual Studio 2022 Desktop development with C++
- download [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) or a later version
