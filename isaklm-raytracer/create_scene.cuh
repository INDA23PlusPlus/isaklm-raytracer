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
#include "create_models.cuh"
#include "create_kd_tree.cuh"


Scene create_scene()
{
	std::vector<Triangle> triangles = create_models();


	int triangle_bytes = triangles.size() * sizeof(Triangle);

	Triangle* host_triangles = (Triangle*)malloc(triangle_bytes);

	for (int i = 0; i < triangles.size(); ++i)
	{
		host_triangles[i] = triangles[i];
	}

	Triangle* device_triangles;
	cudaMalloc(&device_triangles, triangle_bytes);
	cudaMemcpy(device_triangles, host_triangles, triangle_bytes, cudaMemcpyHostToDevice);
	free(host_triangles);

	std::cout << "triangle count: " << triangles.size() << '\n';


	std::vector<int> light_indicies;

	for (int i = 0; i < triangles.size(); ++i)
	{
		Vec3D emittance = triangles[i].material.emittance;

		if (emittance.x > 0 || emittance.y > 0 || emittance.z > 0)
		{
			light_indicies.push_back(i);
		}
	}

	int light_bytes = light_indicies.size() * sizeof(int);

	int* host_light_indicies = (int*)malloc(light_bytes);

	for (int i = 0; i < light_indicies.size(); ++i)
	{
		host_light_indicies[i] = light_indicies[i];
	}

	int* device_light_indicies;
	cudaMalloc(&device_light_indicies, light_bytes);
	cudaMemcpy(device_light_indicies, host_light_indicies, light_bytes, cudaMemcpyHostToDevice);
	free(host_light_indicies);

	std::cout << "light count: " << light_indicies.size() << '\n';


	Scene scene = { device_triangles, int(triangles.size()), device_light_indicies, int(light_indicies.size()), create_kd_tree(triangles) };


	return scene;
}