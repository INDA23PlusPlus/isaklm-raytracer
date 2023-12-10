#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>
#include <random>

#include "macros.h"
#include "math_library.cuh"


struct G_Buffer
{
	Vec3D* frame_buffer;
	uint32_t* random_numbers;

	G_Buffer()
	{
		cudaMalloc(&frame_buffer, SCREEN_W * SCREEN_H * sizeof(Vec3D));


		uint32_t* initial_seeds = (uint32_t*)malloc(SCREEN_W * SCREEN_H * sizeof(uint32_t));

		std::mt19937 generator;
		std::uniform_int_distribution<uint32_t> distribution(0, UINT32_MAX);

		for (int i = 0; i < SCREEN_W * SCREEN_H; ++i)
		{
			initial_seeds[i] = distribution(generator);
		}

		cudaMalloc(&random_numbers, SCREEN_W * SCREEN_H * sizeof(uint32_t));
		cudaMemcpy(random_numbers, initial_seeds, SCREEN_W * SCREEN_H * sizeof(uint32_t), cudaMemcpyHostToDevice);
	}
};