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


struct Material
{
	Vec3D albedo;
	Vec3D emittance;
};

struct Triangle
{
	Vec3D p1, p2, p3; // points
	Vec2D t1, t2, t3; // texture coordinates
	Vec3D normal, tangent, bitangent;
	Material material;
};

struct Texture
{
	uchar4* buffer;
	int width = 0;
	int height = 0;
};

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

struct Scene
{
	Triangle* triangles;
	int triangle_count;
	Texture sky_texture;
};

void load_mesh(std::string file_path, std::vector<Triangle>& triangles, Vec3D offset, float scale, Material material)
{
	int vertex_count = 0;
	int face_count = 0;


	std::ifstream file;
	file.open(file_path);

	std::string line = "";


	while (getline(file, line))
	{
		if (line[0] == 'v')
		{
			++vertex_count;
		}
		else if (line[0] == 'f')
		{
			++face_count;
		}
	}


	std::vector<Vec3D> verticies(vertex_count);
	std::vector<Triangle> mesh(face_count);


	file.close();
	file.open(file_path);


	int vertex_index = 0;
	int face_index = 0;

	while (getline(file, line))
	{
		std::istringstream string_stream(line);

		char begin;


		if (line[0] == 'v')
		{
			Vec3D vertex = ZERO_VEC3D;

			string_stream >> begin >> vertex.x >> vertex.y >> vertex.z;


			verticies[vertex_index] = vertex * scale + offset;


			++vertex_index;
		}
		else if (line[0] == 'f')
		{
			Triangle triangle;

			int vertex1 = 0;
			int vertex2 = 0;
			int vertex3 = 0;

			string_stream >> begin >> vertex1 >> vertex2 >> vertex3;

			triangle.p1 = verticies[vertex1 - 1];
			triangle.p2 = verticies[vertex2 - 1];
			triangle.p3 = verticies[vertex3 - 1];


			triangle.normal = normalize(cross(triangle.p2 - triangle.p1, triangle.p3 - triangle.p1));
			triangle.tangent = normalize(triangle.p2 - triangle.p1);
			triangle.bitangent = normalize(cross(triangle.normal, triangle.tangent));


			triangle.t1 = ZERO_VEC2D;
			triangle.t2 = ZERO_VEC2D;
			triangle.t3 = ZERO_VEC2D;


			triangle.material = material;


			mesh[face_index] = triangle;


			++face_index;
		}
	}


	int prior_size = triangles.size();

	triangles.resize(prior_size + mesh.size());


	for (int i = 0; i < mesh.size(); ++i)
	{
		triangles[i + prior_size] = mesh[i];
	}
}

Scene create_scene()
{
	std::vector<Triangle> triangles(0);

	Material plane_material = { { 0.8f, 0.85f, 0.8f }, ZERO_VEC3D };

	Triangle plane_triangle1 = { { -50.0f, 0.0f, -50.0f }, { -50.0f, 0.0f, 50.0f }, { 50.0f, 0.0f, 50.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, plane_material };
	Triangle plane_triangle2 = { { -50.0f, 0.0f, -50.0f }, { 50.0f, 0.0f, 50.0f }, { 50.0f, 0.0f, -50.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, plane_material };

	triangles.push_back(plane_triangle1);
	triangles.push_back(plane_triangle2);

	Material light_material = { { 0.4f, 0.4f, 0.4f }, { 10.0f, 10.0f, 10.0f } };

	Triangle light_triangle1 = { { -4.0f, 0.0f, 7.0f }, { -4.0f, 8.0f, 7.0f }, { 4.0f, 8.0f, 7.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, light_material };
	Triangle light_triangle2 = { { -4.0f, 0.0f, 7.0f }, { 4.0f, 8.0f, 7.0f }, { 4.0f, 0.0f, 7.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, light_material };

	triangles.push_back(light_triangle1);
	triangles.push_back(light_triangle2);


	Material tea_pot_material = { { 0.3f, 0.8f, 0.9f }, ZERO_VEC3D };

	load_mesh("models/teapot.obj", triangles, { -3.0f, 0, 4.0f }, 1.0f, tea_pot_material);

	Material cheburashka_material = { { 0.9f, 0.8f, 0.3f }, ZERO_VEC3D };

	load_mesh("models/cheburashka.obj", triangles, { 0.5f, -0.4f, 0.0f }, 4.0f, cheburashka_material);


	int triangle_bytes = triangles.size() * sizeof(Triangle);

	Triangle* host_triangles = (Triangle*)malloc(triangle_bytes);

	for (int i = 0; i < triangles.size(); ++i)
	{
		host_triangles[i] = triangles[i];
	}

	std::cout << "triangle count: " << triangles.size() << '\n';



	Triangle* device_triangles;

	cudaMalloc(&device_triangles, triangle_bytes);

	cudaMemcpy(device_triangles, host_triangles, triangle_bytes, cudaMemcpyHostToDevice);

	free(host_triangles);


	Texture sky_texture;
	make_texture(sky_texture, "textures/sky.jpg");


	return { device_triangles, int(triangles.size()), sky_texture };
}