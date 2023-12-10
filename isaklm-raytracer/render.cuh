#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "macros.h"
#include "scene.cuh"
#include "camera.cuh"
#include "screen.cuh"
#include "math_library.cuh"
#include "path_tracing.cuh"


__global__ void reset_frame(Vec3D* frame_buffer)
{
    const int screen_cell_x = (blockIdx.x * blockDim.x + threadIdx.x) * SCREEN_CELL_W;
    const int screen_cell_y = (blockIdx.y * blockDim.y + threadIdx.y) * SCREEN_CELL_H;

    for (int y = screen_cell_y; y < screen_cell_y + SCREEN_CELL_H; ++y)
    {
        for (int x = screen_cell_x; x < screen_cell_x + SCREEN_CELL_W; ++x)
        {
            int pixel_index = y * SCREEN_W + x;

            frame_buffer[pixel_index] = ZERO_VEC3D;
        }
    }
}


__global__ void draw_frame(cudaSurfaceObject_t screen_cuda_surface_object, Vec3D* frame_buffer, int sample_count)
{
    const int screen_cell_x = (blockIdx.x * blockDim.x + threadIdx.x) * SCREEN_CELL_W;
    const int screen_cell_y = (blockIdx.y * blockDim.y + threadIdx.y) * SCREEN_CELL_H;

    for (int y = screen_cell_y; y < screen_cell_y + SCREEN_CELL_H; ++y)
    {
        for (int x = screen_cell_x; x < screen_cell_x + SCREEN_CELL_W; ++x)
        {
            int pixel_index = y * SCREEN_W + x;

            Vec3D color = frame_buffer[pixel_index] * (1.0f / sample_count);


            color = gamma_correction(color);

            color.x = clamp(color.x, 0, 1.0f);
            color.y = clamp(color.y, 0, 1.0f);
            color.z = clamp(color.z, 0, 1.0f);



            uchar4 displayed_color = make_uchar4(color.x * MAX_COLOR_CHANNEL, color.y * MAX_COLOR_CHANNEL, color.z * MAX_COLOR_CHANNEL, MAX_COLOR_CHANNEL);


            surf2Dwrite(displayed_color, screen_cuda_surface_object, x * sizeof(uchar4), y);
        }
    }
}


inline void render(cudaSurfaceObject_t screen_cuda_surface_object, G_Buffer g_buffer, Scene scene, Camera camera, int sample_count)
{
    dim3 grid(20, 45, 1);
    dim3 block(32, 8, 1);


    if (sample_count == 1)
    {
        reset_frame<<<grid, block>>>(g_buffer.frame_buffer);
    }
    
    path_tracing<<<grid, block>>>(g_buffer, scene, camera);

    draw_frame<<<grid, block>>>(screen_cuda_surface_object, g_buffer.frame_buffer, sample_count);
}

