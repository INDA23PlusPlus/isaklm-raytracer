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
#include "scene.cuh"
#include "mesh_loading.cuh"
#include "create_kd_tree.cuh"


Scene create_scene()
{
	std::vector<Triangle> triangles(0);


	Material plane_material = { { 0.8f, 0.85f, 0.8f }, ZERO_VEC3D };

	Triangle plane_triangle1 = { { -50.0f, 0.0f, -50.0f }, { -50.0f, 0.0f, 50.0f }, { 50.0f, 0.0f, 50.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, plane_material };
	Triangle plane_triangle2 = { { -50.0f, 0.0f, -50.0f }, { 50.0f, 0.0f, 50.0f }, { 50.0f, 0.0f, -50.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, plane_material };

	triangles.push_back(plane_triangle1);
	triangles.push_back(plane_triangle2);


	Material light_material = { { 0.4f, 0.4f, 0.4f }, { 5.0f, 5.0f, 5.0f } };

	Triangle light_triangle1 = { { -4.0f, 0.0f, 7.0f }, { -4.0f, 8.0f, 7.0f }, { 4.0f, 8.0f, 7.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, light_material };
	Triangle light_triangle2 = { { -4.0f, 0.0f, 7.0f }, { 4.0f, 8.0f, 7.0f }, { 4.0f, 0.0f, 7.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, light_material };

	triangles.push_back(light_triangle1);
	triangles.push_back(light_triangle2);


	Material tea_pot_material = { { 0.3f, 0.8f, 0.9f }, ZERO_VEC3D };

	load_mesh("models/teapot.obj", triangles, { -3.0f, 0.0f, 4.0f }, 1.0f, tea_pot_material, true);

	Material cheburashka_material = { { 0.9f, 0.8f, 0.3f }, ZERO_VEC3D };

	load_mesh("models/cheburashka.obj", triangles, { 0.5f, -0.35f, 0.0f }, 4.0f, cheburashka_material, true);


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


	Scene scene = { device_triangles, int(triangles.size()), sky_texture, create_kd_tree(triangles) };


	return scene;
}