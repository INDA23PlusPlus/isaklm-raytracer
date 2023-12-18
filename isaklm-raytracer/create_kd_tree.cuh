#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <algorithm>

#include "macros.h"
#include "math_library.cuh"
#include "scene.cuh"


Bounding_Box get_bounding_box(std::vector<int>& triangle_indicies, std::vector<Triangle>& triangles)
{
	float epsilon = 0.01f;


	Bounding_Box bounding_box = { { FLT_MAX, FLT_MAX, FLT_MAX }, -Vec3D{ FLT_MAX, FLT_MAX, FLT_MAX } };


	for (int i = 0; i < triangle_indicies.size(); ++i)
	{
		Triangle triangle = triangles[triangle_indicies[i]];


		Bounding_Box triangle_bounding_box;

		triangle_bounding_box.min.x = fminf(triangle.p1.x, fminf(triangle.p2.x, triangle.p3.x));
		triangle_bounding_box.min.y = fminf(triangle.p1.y, fminf(triangle.p2.y, triangle.p3.y));
		triangle_bounding_box.min.z = fminf(triangle.p1.z, fminf(triangle.p2.z, triangle.p3.z));

		triangle_bounding_box.max.x = fmaxf(triangle.p1.x, fmaxf(triangle.p2.x, triangle.p3.x));
		triangle_bounding_box.max.y = fmaxf(triangle.p1.y, fmaxf(triangle.p2.y, triangle.p3.y));
		triangle_bounding_box.max.z = fmaxf(triangle.p1.z, fmaxf(triangle.p2.z, triangle.p3.z));


		bounding_box.min.x = fminf(triangle_bounding_box.min.x, bounding_box.min.x);
		bounding_box.min.y = fminf(triangle_bounding_box.min.y, bounding_box.min.y);
		bounding_box.min.z = fminf(triangle_bounding_box.min.z, bounding_box.min.z);

		bounding_box.max.x = fmaxf(triangle_bounding_box.max.x, bounding_box.max.x);
		bounding_box.max.y = fmaxf(triangle_bounding_box.max.y, bounding_box.max.y);
		bounding_box.max.z = fmaxf(triangle_bounding_box.max.z, bounding_box.max.z);
	}


	bounding_box.min -= { epsilon, epsilon, epsilon };
	bounding_box.max += { epsilon, epsilon, epsilon };


	return bounding_box;
}

bool triangle_behind_plane(Triangle triangle, int plane_axis, float plane_offset)
{
	if (plane_axis == 0)
	{
		float min_x = fminf(triangle.p1.x, fminf(triangle.p2.x, triangle.p3.x));

		if (min_x <= plane_offset)
		{
			return true;
		}
	}
	else if (plane_axis == 1)
	{
		float min_y = fminf(triangle.p1.y, fminf(triangle.p2.y, triangle.p3.y));

		if (min_y <= plane_offset)
		{
			return true;
		}
	}
	else
	{
		float min_z = fminf(triangle.p1.z, fminf(triangle.p2.z, triangle.p3.z));

		if (min_z <= plane_offset)
		{
			return true;
		}
	}

	return false;
}

bool triangle_afore_plane(Triangle triangle, int plane_axis, float plane_offset)
{
	if (plane_axis == 0)
	{
		float max_x = fmaxf(triangle.p1.x, fmaxf(triangle.p2.x, triangle.p3.x));

		if (max_x >= plane_offset)
		{
			return true;
		}
	}
	else if (plane_axis == 1)
	{
		float max_y = fmaxf(triangle.p1.y, fmaxf(triangle.p2.y, triangle.p3.y));

		if (max_y >= plane_offset)
		{
			return true;
		}
	}
	else
	{
		float max_z = fmaxf(triangle.p1.z, fmaxf(triangle.p2.z, triangle.p3.z));

		if (max_z >= plane_offset)
		{
			return true;
		}
	}

	return false;
}

float get_plane_offset(std::vector<int>& node_triangle_indicies, std::vector<Triangle>& triangles, int plane_axis)
{
	std::vector<float> triangle_values(node_triangle_indicies.size(), 0);

	for (int i = 0; i < node_triangle_indicies.size(); ++i)
	{
		Triangle triangle = triangles[node_triangle_indicies[i]];

		if (plane_axis == 0)
		{
			float min = fminf(triangle.p1.x, fminf(triangle.p2.x, triangle.p3.x));
			float max = fmaxf(triangle.p1.x, fmaxf(triangle.p2.x, triangle.p3.x));

			triangle_values[i] = (min + max) * 0.5f;
		}
		else if (plane_axis == 1)
		{
			float min = fminf(triangle.p1.y, fminf(triangle.p2.y, triangle.p3.y));
			float max = fmaxf(triangle.p1.y, fmaxf(triangle.p2.y, triangle.p3.y));

			triangle_values[i] = (min + max) * 0.5f;
		}
		else
		{
			float min = fminf(triangle.p1.z, fminf(triangle.p2.z, triangle.p3.z));
			float max = fmaxf(triangle.p1.z, fmaxf(triangle.p2.z, triangle.p3.z));

			triangle_values[i] = (min + max) * 0.5f;
		}
	}

	std::sort(triangle_values.begin(), triangle_values.end());


	return triangle_values[triangle_values.size() / 2];
}

void add_child_nodes(KD_Tree_Node& parent_node, std::vector<int>& node_triangle_indicies, std::list<KD_Tree_Node>& kd_tree_nodes, std::list<int>& triangle_indicies, std::vector<Triangle>& triangles, int depth)
{
	int plane_axis = depth % 3;


	float plane_offset = get_plane_offset(node_triangle_indicies, triangles, plane_axis);

	parent_node.plane_axis = plane_axis;
	parent_node.plane_offset = plane_offset;


	int triangle_count1 = 0;
	int triangle_count2 = 0;

	for (int i = 0; i < node_triangle_indicies.size(); ++i)
	{
		Triangle triangle = triangles[node_triangle_indicies[i]];

		if (triangle_behind_plane(triangle, plane_axis, plane_offset))
		{
			++triangle_count1;
		}

		if (triangle_afore_plane(triangle, plane_axis, plane_offset))
		{
			++triangle_count2;
		}
	}


	std::vector<int> child_triangle_indicies1(triangle_count1);
	std::vector<int> child_triangle_indicies2(triangle_count2);

	{
		int child_triangle_index1 = 0;
		int child_triangle_index2 = 0;

		for (int i = 0; i < node_triangle_indicies.size(); ++i)
		{
			int triangle_index = node_triangle_indicies[i];

			Triangle triangle = triangles[triangle_index];

			if (triangle_behind_plane(triangle, plane_axis, plane_offset))
			{
				child_triangle_indicies1[child_triangle_index1] = triangle_index;

				++child_triangle_index1;
			}

			if (triangle_afore_plane(triangle, plane_axis, plane_offset))
			{
				child_triangle_indicies2[child_triangle_index2] = triangle_index;

				++child_triangle_index2;
			}
		}
	}


	int min_triangle_count = 7;


	if (triangle_count1 > min_triangle_count && depth < KD_TREE_DEPTH)
	{
		parent_node.child_index1 = kd_tree_nodes.size();

		kd_tree_nodes.push_back({ 0, 0, 0, 0.0f, false });

		add_child_nodes(kd_tree_nodes.back(), child_triangle_indicies1, kd_tree_nodes, triangle_indicies, triangles, depth + 1);
	}
	else
	{
		parent_node.child_index1 = kd_tree_nodes.size();

		kd_tree_nodes.push_back({ int(triangle_indicies.size()), triangle_count1, 0, 0.0f, true });

		for (int i = 0; i < triangle_count1; ++i)
		{
			triangle_indicies.push_back(child_triangle_indicies1[i]);
		}
	}


	if (triangle_count2 > min_triangle_count && depth < KD_TREE_DEPTH)
	{
		parent_node.child_index2 = kd_tree_nodes.size();

		kd_tree_nodes.push_back({ 0, 0, 0, 0.0f, false });

		add_child_nodes(kd_tree_nodes.back(), child_triangle_indicies2, kd_tree_nodes, triangle_indicies, triangles, depth + 1);
	}
	else
	{
		parent_node.child_index2 = kd_tree_nodes.size();

		kd_tree_nodes.push_back({ int(triangle_indicies.size()), triangle_count2, 0, 0.0f, true });

		for (int i = 0; i < triangle_count2; ++i)
		{
			triangle_indicies.push_back(child_triangle_indicies2[i]);
		}
	}
}

KD_Tree create_kd_tree(std::vector<Triangle>& triangles)
{
	std::list<KD_Tree_Node> kd_tree_nodes(0);
	std::list<int> triangle_indicies(0);


	kd_tree_nodes.push_back({ 0, 0, 0, 0.0f, false });

	std::vector<int> root_triangle_indicies(triangles.size());

	for (int i = 0; i < triangles.size(); ++i)
	{
		root_triangle_indicies[i] = i;
	}


	add_child_nodes(kd_tree_nodes.back(), root_triangle_indicies, kd_tree_nodes, triangle_indicies, triangles, 0);


	int node_bytes = kd_tree_nodes.size() * sizeof(KD_Tree_Node);
	KD_Tree_Node* host_nodes = (KD_Tree_Node*)malloc(node_bytes);

	{
		int i = 0;

		for (const KD_Tree_Node& node : kd_tree_nodes)
		{
			host_nodes[i] = node;

			++i;
		}
	}

	int index_bytes = triangle_indicies.size() * sizeof(int);
	int* host_triangle_indicies = (int*)malloc(index_bytes);

	{
		int i = 0;

		for (const int& index : triangle_indicies)
		{
			host_triangle_indicies[i] = index;

			++i;
		}
	}

	std::cout << "node count: " << kd_tree_nodes.size() << '\n';
	std::cout << "triangle index count: " << triangle_indicies.size() << '\n';


	KD_Tree_Node* device_nodes;
	cudaMalloc(&device_nodes, node_bytes);
	cudaMemcpy(device_nodes, host_nodes, node_bytes, cudaMemcpyHostToDevice);

	int* device_triangle_indicies;
	cudaMalloc(&device_triangle_indicies, index_bytes);
	cudaMemcpy(device_triangle_indicies, host_triangle_indicies, index_bytes, cudaMemcpyHostToDevice);
	

	return { get_bounding_box(root_triangle_indicies, triangles), device_nodes, device_triangle_indicies };
}