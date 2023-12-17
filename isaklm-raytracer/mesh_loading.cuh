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


void load_mesh(std::string file_path, std::vector<Triangle>& triangles, Vec3D offset, float scale, Material material, bool smooth_normals)
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


	std::vector<Vec3D> vertex_coordinates(vertex_count);
	std::vector<Vec3D> vertex_normals(vertex_count);
	std::vector<Triangle> mesh(face_count);
	std::vector<Int3> triangle_vertex_indicies(face_count);


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


			vertex_coordinates[vertex_index] = vertex * scale + offset;


			++vertex_index;
		}
		else if (line[0] == 'f')
		{
			Triangle triangle;

			int vertex1 = 0;
			int vertex2 = 0;
			int vertex3 = 0;

			string_stream >> begin >> vertex1 >> vertex2 >> vertex3;

			vertex1 -= 1;
			vertex2 -= 1;
			vertex3 -= 1;

			triangle.p1 = vertex_coordinates[vertex1];
			triangle.p2 = vertex_coordinates[vertex2];
			triangle.p3 = vertex_coordinates[vertex3];


			Vec3D normal = normalize(cross(triangle.p2 - triangle.p1, triangle.p3 - triangle.p1));

			if (smooth_normals)
			{
				vertex_normals[vertex1] += normal;
				vertex_normals[vertex2] += normal;
				vertex_normals[vertex3] += normal;

				triangle_vertex_indicies[face_index] = { vertex1, vertex2, vertex3 };
			}
			else
			{
				triangle.n1 = normal;
				triangle.n2 = normal;
				triangle.n3 = normal;
			}


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
		Triangle triangle = mesh[i];

		if (smooth_normals)
		{
			Int3 vertex_indicies = triangle_vertex_indicies[i];

			triangle.n1 = normalize(vertex_normals[vertex_indicies.x]);
			triangle.n2 = normalize(vertex_normals[vertex_indicies.y]);
			triangle.n3 = normalize(vertex_normals[vertex_indicies.z]);
		}

		triangles[i + prior_size] = triangle;
	}
}