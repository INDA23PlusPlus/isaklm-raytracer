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
#include "trace_ray.cuh"


__device__ float get_random_unilateral(G_Buffer g_buffer, int pixel_index)
{
    uint32_t state = g_buffer.random_numbers[pixel_index] * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint32_t random_unsigned_int = (word >> 22u) ^ word;

    g_buffer.random_numbers[pixel_index] = random_unsigned_int;

    return float(random_unsigned_int) / UINT32_MAX;
}

__device__ Vec3D diffuse_direction(G_Buffer g_buffer, int pixel_index, Vec3D normal, Vec3D tangent, Vec3D bitangent)
{
    float random_phi = get_random_unilateral(g_buffer, pixel_index) * TAU;

    float sin_phi = sinf(random_phi);
    float cos_phi = cosf(random_phi);


    float random_unilateral = get_random_unilateral(g_buffer, pixel_index);

    float sqrt_random_unilateral = sqrtf(random_unilateral);


    return sqrt_random_unilateral * cos_phi * tangent + sqrtf(1.0f - random_unilateral) * normal + sqrt_random_unilateral * sin_phi * bitangent;
}


__device__ void trace_path(G_Buffer g_buffer, Ray ray, Scene scene, int pixel_index)
{
    Vec3D outgoing_light = ZERO_VEC3D;

    Vec3D throughput = Vec3D{ 1.0f, 1.0f, 1.0f };


    Sample sample;

    for(int i = 0; i < MAX_BOUNCES; ++i)
    {
        if (trace_ray(ray, scene, sample))
        {
            outgoing_light += sample.material.emittance * throughput;

            throughput *= sample.material.albedo;

            ray.position = sample.position;
            ray.direction = diffuse_direction(g_buffer, pixel_index, sample.normal, sample.tangent, sample.bitangent);
        }
        else
        {
            break;
        }
    }


    g_buffer.frame_buffer[pixel_index] += outgoing_light;
}

__device__ Vec3D random_point_in_pinhole(Camera camera, G_Buffer g_buffer, int pixel_index)
{
    float random_offset_x = (get_random_unilateral(g_buffer, pixel_index) - 0.5f) * camera.pinhole_width;
    float random_offset_y = (get_random_unilateral(g_buffer, pixel_index) - 0.5f) * camera.pinhole_width;

    return camera.position + camera.rotation() * Vec3D { random_offset_x, 0.0f, 0.0f } + camera.rotation() * Vec3D { 0.0f, random_offset_y, 0.0f };
}

__global__ void path_tracing(G_Buffer g_buffer, Scene scene, Camera camera)
{
    const int screen_cell_x = (blockIdx.x * blockDim.x + threadIdx.x) * SCREEN_CELL_W;
    const int screen_cell_y = (blockIdx.y * blockDim.y + threadIdx.y) * SCREEN_CELL_H;

    for (int y = screen_cell_y; y < screen_cell_y + SCREEN_CELL_H; ++y)
    {
        for (int x = screen_cell_x; x < screen_cell_x + SCREEN_CELL_W; ++x)
        {
            int pixel_index = y * SCREEN_W + x;


            float tan_half_FOV = tanf(camera.FOV / 2);

            float random_offset_x = get_random_unilateral(g_buffer, pixel_index);
            float random_offset_y = get_random_unilateral(g_buffer, pixel_index);

            Vec3D ray_direction = normalize({ tan_half_FOV * (x + random_offset_x - SCREEN_W / 2) / float(SCREEN_W / 2), tan_half_FOV * (y + random_offset_y - SCREEN_H / 2) / float(SCREEN_W / 2), 1.0f});

            ray_direction = camera.rotation() * ray_direction;


            trace_path(g_buffer, { random_point_in_pinhole(camera, g_buffer, pixel_index), ray_direction }, scene, pixel_index);
        }
    }
}