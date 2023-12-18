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

	{
		Material floor_material = { { 0.8f, 0.85f, 0.8f }, ZERO_VEC3D, 0.1f, 3.0f };

		Triangle floor_triangle1 = { { -4.0f, 0.0f, -4.0f }, { -4.0f, 0.0f, 4.0f }, { 4.0f, 0.0f, 4.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, floor_material };
		Triangle floor_triangle2 = { { -4.0f, 0.0f, -4.0f }, { 4.0f, 0.0f, 4.0f }, { 4.0f, 0.0f, -4.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, floor_material };

		triangles.push_back(floor_triangle1);
		triangles.push_back(floor_triangle2);
	}

	{
		Material wall_material = { { 0.5f, 0.5f, 0.5f }, ZERO_VEC3D, 0.02f, 7.0f };

		Triangle wall_triangle1 = { { -4.0f, 0.0f, 4.0f }, { -4.0f, 4.0f, 4.0f }, { 4.0f, 4.0f, 4.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };
		Triangle wall_triangle2 = { { -4.0f, 0.0f, 4.0f }, { 4.0f, 4.0f, 4.0f }, { 4.0f, 0.0f, 4.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, -1.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };

		triangles.push_back(wall_triangle1);
		triangles.push_back(wall_triangle2);
	}

	{
		Material wall_material = { { 0.8f, 0.2f, 0.4f }, ZERO_VEC3D, 0.6f, 1.2f };

		Triangle wall_triangle1 = { { -4.0f, 0.0f, 4.0f }, { -4.0f, 4.0f, 4.0f }, { 4.0f, 4.0f, 4.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };
		Triangle wall_triangle2 = { { -4.0f, 0.0f, 4.0f }, { 4.0f, 4.0f, 4.0f }, { 4.0f, 0.0f, 4.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };

		triangles.push_back(wall_triangle1);
		triangles.push_back(wall_triangle2);
	}

	{
		Material wall_material = { { 0.4f, 0.2f, 0.8f }, ZERO_VEC3D, 0.6f, 1.2f };

		Triangle wall_triangle1 = { { -4.0f, 0.0f, -4.0f }, { -4.0f, 4.0f, -4.0f }, { -4.0f, 4.0f, 4.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };
		Triangle wall_triangle2 = { { -4.0f, 0.0f, -4.0f }, { -4.0f, 4.0f, 4.0f }, { -4.0f, 0.0f, 4.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };

		triangles.push_back(wall_triangle1);
		triangles.push_back(wall_triangle2);
	}

	{
		Material wall_material = { { 0.2f, 0.8f, 0.4f }, ZERO_VEC3D, 0.6f, 1.2f };

		Triangle wall_triangle1 = { { 4.0f, 0.0f, -4.0f }, { 4.0f, 4.0f, -4.0f }, { 4.0f, 4.0f, 4.0f }, { -1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };
		Triangle wall_triangle2 = { { 4.0f, 0.0f, -4.0f }, { 4.0f, 4.0f, 4.0f }, { 4.0f, 0.0f, 4.0f }, { -1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, wall_material };

		triangles.push_back(wall_triangle1);
		triangles.push_back(wall_triangle2);
	}

	{
		Material ceiling_material = { { 0.85f, 0.85f, 0.8f }, ZERO_VEC3D, 0.8f, 1.2f };

		Triangle ceiling_triangle1 = { { -4.0f, 4.0f, -4.0f }, { -4.0f, 4.0f, 4.0f }, { 4.0f, 4.0f, 4.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, ceiling_material };
		Triangle ceiling_triangle2 = { { -4.0f, 4.0f, -4.0f }, { 4.0f, 4.0f, 4.0f }, { 4.0f, 4.0f, -4.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, ceiling_material };

		triangles.push_back(ceiling_triangle1);
		triangles.push_back(ceiling_triangle2);
	}

	{
		Material light_material = { { 0.4f, 0.4f, 0.4f }, { 5.0f, 4.5f, 4.0f }, 0.7f, 1.2f };

		Triangle light_triangle1 = { { -1.0f, 3.95f, -1.0f }, { -1.0f, 3.95f, 1.0f }, { 1.0f, 3.95f, 1.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, light_material };
		Triangle light_triangle2 = { { -1.0f, 3.95f, -1.0f }, { 1.0f, 3.95f, 1.0f }, { 1.0f, 3.95f, -1.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, ZERO_VEC2D, ZERO_VEC2D, ZERO_VEC2D, light_material };

		triangles.push_back(light_triangle1);
		triangles.push_back(light_triangle2);
	}


	Material horse_material = { { 0.3f, 0.8f, 0.9f }, ZERO_VEC3D, 0.2f, 2.7f };

	load_mesh("models/horse.obj", triangles, { -1.5f, 0.9f, 0.5f }, rotation_matrix(-1.0f, -HALF_PI) * 12.0f, horse_material, true);

	Material cheburashka_material = { { 0.9f, 0.8f, 0.3f }, ZERO_VEC3D, 0.3f, 2.5f };

	load_mesh("models/cheburashka.obj", triangles, { 3.0f, -0.2f, 1.5f }, rotation_matrix(3.5f) * 2.0f, cheburashka_material, true);

	Material buddha_material = { { 0.9f, 0.2f, 0.5f }, ZERO_VEC3D, 0.1f, 3.2f };

	load_mesh("models/happy_buddha.obj", triangles, { 0.0f, -0.5f, 0.0f }, rotation_matrix(PI) * 8.0f, buddha_material, true);


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