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

__device__ float fresnel_dielectric(Vec3D incoming_direction, Vec3D half_vector, float n1, float n2)
{
    float c = fabsf(dot(incoming_direction, half_vector));

    float g = sqrtf(fmaxf(square(n2) / square(n1) - 1.0f + square(c), 0.0f));


    float factor1 = 0.5f * square((g - c) / (g + c));

    float factor2 = 1.0f + square((c * (g + c) - 1.0f) / (c * (g - c) + 1.0f));


    return factor1 * factor2;
}

__device__ float fresnel_conductor(Vec3D incoming_direction, Vec3D half_vector, float n, float k)
{
    // reference https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/#more-1921

    float n2 = n * n;
    float k2 = k * k;

    float cos_theta = dot(incoming_direction, half_vector);
    float cos_theta2 = square(cos_theta);
    float sin_theta2 = 1.0f - cos_theta2;

    float t0 = n2 - k2 - sin_theta2;
    float a2_plus_b2 = sqrtf(square(t0) + 4.0f * n2 * k2);
    float a = sqrtf(0.5f * (a2_plus_b2 + t0));

    float t1 = a2_plus_b2 + cos_theta2;
    float t2 = 2.0f * a * cos_theta;

    float rs = (t1 - t2) / (t1 + t2);

    float t3 = cos_theta2 * a2_plus_b2 * square(sin_theta2);
    float t4 = t2 * sin_theta2;
    float rp = rs * (t3 - t4) / (t3 + t4);

    return (rs + rp) * 0.5f;
}

__device__ Vec3D microfacet_normal(G_Buffer g_buffer, int pixel_index, Vec3D normal, Vec3D tangent, Vec3D bitangent, float roughness) // GGX distribution
{
    double random_unilateral = get_random_unilateral(g_buffer, pixel_index);


    float cos_theta = sqrtf((1.0f - random_unilateral) / (random_unilateral * (roughness * roughness - 1.0f) + 1.0f));
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    float random_phi = get_random_unilateral(g_buffer, pixel_index) * TAU;

    float cos_phi = cosf(random_phi);
    float sin_phi = sinf(random_phi);


    return tangent * sin_theta * cos_phi + normal * cos_theta + bitangent * sin_theta * sin_phi;
}

__device__ float lambda(Vec3D direction, Vec3D normal, float roughness)
{
    float direction_dot_normal = dot(direction, normal);
    float direction_dot_normal2 = square(direction_dot_normal);
    float tan2 = (1 - direction_dot_normal2) / direction_dot_normal2;

    return (sqrtf(1.0f + square(roughness) + tan2) - 1.0f) * 0.5f;
}

__device__ Vec3D specular_weight(Vec3D incoming_direction, Vec3D outgoing_direction, Vec3D half_vector, Vec3D normal, float roughness)
{
    float g = 1.0f / (1.0f + lambda(incoming_direction, normal, roughness) + lambda(outgoing_direction, normal, roughness));

    float weight = fabsf(dot(incoming_direction, half_vector)) * g / (fabsf(dot(normal, half_vector) * fabsf(dot(incoming_direction, normal))));

    return { weight, weight, weight };
}

__device__ Vec3D specular_direction(Vec3D incoming_direction, Vec3D half_vector)
{
    return 2.0f * dot(incoming_direction, half_vector) * half_vector - incoming_direction;
}

__device__ Vec3D refraction_direction(Vec3D incoming_direction, Vec3D half_vector, float n1, float n2)
{
    float c = dot(incoming_direction, half_vector);
    float n = n1 / n2;

    return (n * c - sqrtf(fmaxf(1.0f + n * n * (c * c - 1.0f), 0.0f))) * half_vector - n * incoming_direction;
}

__device__ Vec3D get_scattered_light(Vec3D& ray_direction, bool& inside_medium, G_Buffer g_buffer, int pixel_index, Sample& sample) // returns amount of scattered light
{
    // microfacet models found at: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf

    ray_direction = -ray_direction; // all bsdf equations assume that the direction_vector is pointing away from the surface.


    Vec3D half_vector = microfacet_normal(g_buffer, pixel_index, sample.normal, sample.tangent, sample.bitangent, sample.roughness);


    if (sample.extinction > 0.0f) // metallic reflection
    {
        float fresnel_term = fresnel_conductor(ray_direction, half_vector, sample.refractive_index, sample.extinction);

        Vec3D incoming_direction = ray_direction;
        Vec3D outgoing_direction = specular_direction(incoming_direction, half_vector);

        ray_direction = outgoing_direction;

        return sample.albedo * specular_weight(incoming_direction, outgoing_direction, half_vector, sample.normal, sample.roughness) * fresnel_term;
    }
    else
    {
        float n1 = 1.0f;
        float n2 = sample.refractive_index;

        if (inside_medium)
        {
            n1 = n2;
            n2 = 1.0f;
        }


        float fresnel_term = fresnel_dielectric(ray_direction, half_vector, n1, n2);
        float choose_specular = get_random_unilateral(g_buffer, pixel_index);

        if (choose_specular < fresnel_term) // specular reflection
        {
            Vec3D incoming_direction = ray_direction;
            Vec3D outgoing_direction = specular_direction(incoming_direction, half_vector);

            ray_direction = outgoing_direction;

            Vec3D weight = { 1.0f, 1.0f, 1.0f };

            if (!inside_medium) // since this brdf loses some energy, it makes the glass look too dark due to the many bounces
            {
                weight = specular_weight(incoming_direction, outgoing_direction, half_vector, sample.normal, sample.roughness);
            }

            return weight;
        }
        else if(sample.transparent) // refraction
        {
            inside_medium = !inside_medium;

            Vec3D incoming_direction = ray_direction;
            Vec3D outgoing_direction = refraction_direction(incoming_direction, half_vector, n1, n2);

            ray_direction = outgoing_direction;

            return specular_weight(incoming_direction, outgoing_direction, half_vector, sample.normal, sample.roughness) * sample.albedo;
        }
        else // diffuse reflection
        {
            ray_direction = diffuse_direction(g_buffer, pixel_index, sample.normal, sample.tangent, sample.bitangent);

            return sample.albedo;
        }
    }
}


__device__ void trace_path(G_Buffer g_buffer, Ray ray, Scene scene, int pixel_index)
{
    Vec3D outgoing_light = ZERO_VEC3D;

    Vec3D throughput = Vec3D{ 1.0f, 1.0f, 1.0f };

    bool inside_medium = false;


    while(true)
    {
        Sample sample;

        if (trace_ray(ray, scene, sample))
        {
            outgoing_light += sample.emittance * throughput;

            ray.position = sample.position;


            throughput *= get_scattered_light(ray.direction, inside_medium, g_buffer, pixel_index, sample);
        }
        else
        {
            break;
        }


        float survival_probability = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));

        float random_unilateral = get_random_unilateral(g_buffer, pixel_index);

        if (random_unilateral > survival_probability)
        {
            break;
        }

        throughput *= (1.0f / survival_probability);
    }


    g_buffer.frame_buffer[pixel_index] += outgoing_light;
}

__device__ Vec3D random_point_in_pinhole(Camera camera, G_Buffer g_buffer, int pixel_index)
{
    float random_theta = get_random_unilateral(g_buffer, pixel_index) * TAU;
    float random_r = sqrtf(get_random_unilateral(g_buffer, pixel_index)) * camera.aperture_radius;

    float offset_x = random_r * cosf(random_theta);
    float offset_y = random_r * sinf(random_theta);

    return camera.position + camera.rotation() * Vec3D{ offset_x, 0.0f, 0.0f } + camera.rotation() * Vec3D{ 0.0f, offset_y, 0.0f };
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