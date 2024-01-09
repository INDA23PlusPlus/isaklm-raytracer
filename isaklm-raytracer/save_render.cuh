#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "macros.h"
#include "math_library.cuh"
#include "camera.cuh"
#include "screen.cuh"
#include "render.cuh"
#include "lodepng/lodepng.h"


void encode_image(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
	unsigned error = lodepng::encode(filename, image, width, height);
	
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

void save_render(G_Buffer g_buffer)
{
#define COLOR_CHANNEL_COUNT 4

	std::vector<unsigned char> image(SCREEN_W * SCREEN_H * COLOR_CHANNEL_COUNT, 0);

	int frame_buffer_bytes = SCREEN_W * SCREEN_H * sizeof(Vec3D);
	int sample_count_bytes = SCREEN_W * SCREEN_H * sizeof(int);

	Vec3D* frame_buffer = (Vec3D*)malloc(frame_buffer_bytes);
	int* sample_count = (int*)malloc(sample_count_bytes);

	cudaMemcpy(frame_buffer, g_buffer.frame_buffer, frame_buffer_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(sample_count, g_buffer.sample_count, sample_count_bytes, cudaMemcpyDeviceToHost);


	for (int y = 0; y < SCREEN_H; ++y)
	{
		for (int x = 0; x < SCREEN_W; ++x)
		{
			int pixel_index = y * SCREEN_W + x;

			Vec3D color = frame_buffer[pixel_index] * (1.0f / sample_count[pixel_index]);

			color = correct_color(color);


			uchar4 displayed_color = make_uchar4(color.x * MAX_COLOR_CHANNEL, color.y * MAX_COLOR_CHANNEL, color.z * MAX_COLOR_CHANNEL, MAX_COLOR_CHANNEL);


			int flipped_pixel_index = (SCREEN_H - y - 1) * SCREEN_W + x;

			image[flipped_pixel_index * COLOR_CHANNEL_COUNT + 0] = displayed_color.x;
			image[flipped_pixel_index * COLOR_CHANNEL_COUNT + 1] = displayed_color.y;
			image[flipped_pixel_index * COLOR_CHANNEL_COUNT + 2] = displayed_color.z;
			image[flipped_pixel_index * COLOR_CHANNEL_COUNT + 3] = displayed_color.w;
		}
	}

#undef COLOR_CHANNEL_COUNT

	encode_image("renders/render.png", image, SCREEN_W, SCREEN_H);
}