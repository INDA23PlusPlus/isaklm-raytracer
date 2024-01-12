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


std::vector<Triangle> create_models()
{
	std::vector<Triangle> triangles(0);

	load_mesh("models/room.obj", "materials/room.mat", triangles, { { 0.0f, 1.5f, 0.0f }, rotation_matrix(0.1f) }, false);

	load_mesh("models/desk.obj", "materials/desk.mat", triangles, { { -0.2f, 0.502f, 0.9f }, rotation_matrix(3.24f, -HALF_PI) * 0.013f }, false);

	load_mesh("models/table.obj", "materials/table.mat", triangles, { { 2.05f, 0.42f, -1.1f }, rotation_matrix(-1.5f) * 0.25f }, false);

	load_mesh("models/chair.obj", "materials/chair.mat", triangles, { { -0.15f, 0.67f, -0.25f }, rotation_matrix(-0.1f) * 0.0014f }, false);

	load_mesh("models/simple_chair.obj", "materials/simple_chair.mat", triangles, { { 1.9f, 0.532f, 0.7f }, rotation_matrix(-2.4f) * 1.2f }, false);

	load_mesh("models/outlet.obj", "materials/outlet.mat", triangles, { { -1.6f, 0.4f, 1.665f }, rotation_matrix(PI + 0.1f) * 0.01f }, false);

	load_mesh("models/outlet.obj", "materials/outlet.mat", triangles, { { 0.4f, 0.4f, -1.5446f }, rotation_matrix(0.1f) * 0.01f }, false);

	load_mesh("models/dragon.obj", "materials/dragon.mat", triangles, { { -0.5f, 1.46f, 0.8f }, rotation_matrix(2.1f) * 1.3f }, false);

	load_mesh("models/glass.obj", "materials/glass.mat", triangles, { { -1.25f, 1.127f, 0.7f }, rotation_matrix(2.1f) * 0.025f }, true);

	load_mesh("models/glass.obj", "materials/glass.mat", triangles, { { 0.6f, 1.127f, 0.7f }, rotation_matrix(2.1f) * 0.025f }, true);


	return triangles;
}