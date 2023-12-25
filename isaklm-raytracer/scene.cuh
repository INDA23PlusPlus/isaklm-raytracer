#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "macros.h"
#include "math_library.cuh"
#include "stb_image/stb_image.h"


struct Texture
{
	uchar4* buffer;
	int width = 0;
	int height = 0;
};

#define NO_TEXTURE Texture{ nullptr, 0, 0 }

void make_texture(Texture& texture, const std::string& file_path)
{
	//stbi_set_flip_vertically_on_load(1);

	int bits_per_pixel = 0;

	int texture_channel_count = 4;

	uint8_t* image_buffer = stbi_load(file_path.c_str(), &texture.width, &texture.height, &bits_per_pixel, texture_channel_count);


	int pixel_count = texture.width * texture.height;

	int texture_bytes = pixel_count * sizeof(uchar4);

	uchar4* temporary_buffer = (uchar4*)malloc(texture_bytes);


	for (int i = 0; i < pixel_count; ++i)
	{
		int offset = i * texture_channel_count;

		uint8_t r = image_buffer[offset + 0];
		uint8_t g = image_buffer[offset + 1];
		uint8_t b = image_buffer[offset + 2];
		uint8_t a = image_buffer[offset + 3];

		temporary_buffer[i] = make_uchar4(r, g, b, a);
	}

	free(image_buffer);


	cudaMalloc(&texture.buffer, texture_bytes);
	cudaMemcpy(texture.buffer, temporary_buffer, texture_bytes, cudaMemcpyHostToDevice);


	free(temporary_buffer);
}

struct Material
{
	Vec3D albedo;
	Texture texture;
	Vec3D emittance;
	float roughness;
	float refractive_index;
	float extinction;
	bool transparent;
};

struct Triangle
{
	Vec3D p1, p2, p3; // points
	Vec3D n1, n2, n3;
	Vec2D t1, t2, t3; // texture coordinates
	Material material;
};

struct KD_Tree_Node
{
	union
	{
		int index_offset, child_index1;
	};

	union
	{
		int triangle_count, child_index2;
	};

	uint8_t plane_axis;
	float plane_offset; // offset from origin in the plane acis

	bool is_leaf_node;
};

struct Bounding_Box
{
	Vec3D min, max;
};

struct KD_Tree
{
	Bounding_Box bounding_box;
	KD_Tree_Node* nodes;
	int* triangle_indicies;
};

struct Scene
{
	Triangle* triangles;
	int triangle_count;
	Texture sky_texture;

	KD_Tree kd_tree;
};