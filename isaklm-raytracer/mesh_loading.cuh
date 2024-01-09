#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <set>

#include "macros.h"
#include "math_library.cuh"
#include "scene.cuh"


struct Transformation
{
	Vec3D offset;
	Matrix3X3 matrix;
};

namespace OBJ
{
	struct Vertex
	{
		int position_index;
		int texture_coords_index;
		int normal_index;
	};

	struct Triangle
	{
		Vertex v1, v2, v3;
		std::string material_name;
	};
}


Bounding_Box get_bounding_box(int start_index, std::vector<Triangle>& triangles)
{
	Bounding_Box bounding_box = { { FLT_MAX, FLT_MAX, FLT_MAX }, -Vec3D{ FLT_MAX, FLT_MAX, FLT_MAX } };

	for (int i = start_index; i < triangles.size(); ++i)
	{
		Triangle triangle = triangles[i];

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

	return bounding_box;
}

std::vector<std::string> split_string(std::string string, char delimiter, bool include_empty = false)
{
	std::vector<std::string> strings;

	std::string new_string = "";

	for (int i = 0; i < string.size(); ++i)
	{
		char character = string[i];

		if (character == delimiter)
		{
			if (new_string != "" || include_empty)
			{
				strings.push_back(new_string);
				new_string = "";
			}
		}
		else
		{
			new_string += character;
		}
	}

	if (new_string != "")
	{
		strings.push_back(new_string);
	}

	return strings;
}

OBJ::Vertex create_vertex(std::vector<std::string> vertex_data, int position_count, int texture_coordinate_count, int normal_count)
{
	OBJ::Vertex vertex = { -1, -1, -1 };

	if (vertex_data.size() > 0)
	{
		int index = std::stoi(vertex_data[0]);

		if (index > 0)
		{
			vertex.position_index = index - 1;
		}
		else
		{
			vertex.position_index = position_count + index;
		}
	}
	if (vertex_data.size() > 1 && vertex_data[1] != "")
	{
		int index = std::stoi(vertex_data[1]);

		if (index > 0)
		{
			vertex.texture_coords_index = index - 1;
		}
		else
		{
			vertex.texture_coords_index = texture_coordinate_count + index;
		}
	}
	if (vertex_data.size() > 2)
	{
		int index = std::stoi(vertex_data[2]);

		if (index > 0)
		{
			vertex.normal_index = index - 1;
		}
		else
		{
			vertex.normal_index = normal_count + index;
		}
	}

	return vertex;
}

Material load_material(std::string material_file_path, std::string material_name)
{
	Material material = { ZERO_VEC3D, ZERO_VEC3D, 0.0f, 0.0f, 0.0f, false, NO_TEXTURE };

	bool found_material = false;


	std::ifstream file;
	file.open(material_file_path);

	std::string line = "";


	while (getline(file, line))
	{
		if (line == "material " + material_name)
		{
			found_material = true;
		}
		else if (found_material)
		{
			if (line == "")
			{
				break;
			}

			std::vector<std::string> strings = split_string(line, ' ');

			if (strings[0] == "albedo")
			{
				material.albedo = { std::stof(strings[1]), std::stof(strings[2]), std::stof(strings[3]) };
			}
			else if (strings[0] == "emittance")
			{
				material.emittance = { std::stof(strings[1]), std::stof(strings[2]), std::stof(strings[3]) };
			}
			else if (strings[0] == "roughness")
			{
				material.roughness = std::stof(strings[1]);
			}
			else if (strings[0] == "n")
			{
				material.refractive_index = std::stof(strings[1]);
			}
			else if (strings[0] == "k")
			{
				material.extinction = std::stof(strings[1]);
			}
			else if (strings[0] == "transparent")
			{
				material.transparent = true;
			}
			else if (strings[0] == "texture")
			{
				Texture texture;

				make_texture(texture, strings[1]);

				material.texture = texture;
			}
		}
	}


	file.close();

	return material;
}

void load_mesh(std::string model_file_path, std::string material_file_path, std::vector<Triangle>& triangles, Transformation transformation, bool smooth_normals)
{
	std::vector<Vec3D> vertex_positions;
	std::vector<Vec3D> vertex_normals;
	std::vector<Vec2D> vertex_texture_coords;

	std::vector<OBJ::Triangle> mesh;

	std::map<std::string, Material> materials;


	{
		std::list<Vec3D> positions;
		std::list<Vec3D> normals;
		std::list<Vec2D> texture_coordinates;

		std::string material_name = "";

		std::list<OBJ::Triangle> triangle_list;

		std::set<int> false_normals;


		std::ifstream file;
		file.open(model_file_path);

		std::string line = "";


		while (getline(file, line))
		{
			std::vector<std::string> strings = split_string(line, ' ');

			if (strings.size() > 0)
			{
				if (strings[0] == "v")
				{
					Vec3D position = ZERO_VEC3D;

					position.x = std::stof(strings[1]);
					position.y = std::stof(strings[2]);
					position.z = std::stof(strings[3]);

					positions.push_back(position);
				}
				else if (strings[0] == "vn")
				{
					Vec3D normal = ZERO_VEC3D;

					normal.x = std::stof(strings[1]);
					normal.y = std::stof(strings[2]);
					normal.z = std::stof(strings[3]);

					if (normal.x == 0 && normal.y == 0 && normal.z == 0)
					{
						false_normals.insert(normals.size());
					}

					normals.push_back(normal);
				}
				else if (strings[0] == "vt")
				{
					Vec2D texture_coords = ZERO_VEC2D;

					texture_coords.u = std::stof(strings[1]);
					texture_coords.v = 1.0f - std::stof(strings[2]);

					texture_coordinates.push_back(texture_coords);
				}
				else if (strings[0] == "usemtl")
				{
					material_name = strings[1];

					if (materials.count(material_name) == 0)
					{
						materials[material_name] = load_material(material_file_path, material_name);
					}
				}
				else if (strings[0] == "f")
				{
					OBJ::Vertex v1 = create_vertex(split_string(strings[1], '/', true), positions.size(), texture_coordinates.size(), normals.size());

					if (false_normals.count(v1.normal_index) == 0)
					{
						for (int i = 3; i < strings.size(); ++i)
						{
							OBJ::Vertex v2 = create_vertex(split_string(strings[i - 1], '/', true), positions.size(), texture_coordinates.size(), normals.size());

							OBJ::Vertex v3 = create_vertex(split_string(strings[i], '/', true), positions.size(), texture_coordinates.size(), normals.size());


							triangle_list.push_back({ v1, v2, v3, material_name });
						}
					}
				}
			}
		}


		vertex_positions = { positions.begin(), positions.end() };
		vertex_normals = { normals.begin(), normals.end() };
		vertex_texture_coords = { texture_coordinates.begin(), texture_coordinates.end() };

		mesh = { triangle_list.begin(), triangle_list.end() };
	}


	std::vector<Vec3D> computed_normals(vertex_positions.size(), ZERO_VEC3D);

	for (const OBJ::Triangle& triangle : mesh)
	{
		Vec3D p1 = vertex_positions[triangle.v1.position_index];
		Vec3D p2 = vertex_positions[triangle.v2.position_index];
		Vec3D p3 = vertex_positions[triangle.v3.position_index];

		Vec3D normal = normalize(cross(p2 - p1, p3 - p1));


		computed_normals[triangle.v1.position_index] += normal;
		computed_normals[triangle.v2.position_index] += normal;
		computed_normals[triangle.v3.position_index] += normal;
	}


	int prior_size = triangles.size();

	triangles.resize(prior_size + mesh.size());

	for (int i = 0; i < mesh.size(); ++i)
	{
		OBJ::Triangle obj_triangle = mesh[i];

		Vec3D p1 = vertex_positions[obj_triangle.v1.position_index];
		Vec3D p2 = vertex_positions[obj_triangle.v2.position_index];
		Vec3D p3 = vertex_positions[obj_triangle.v3.position_index];

		Vec3D normal = normalize(cross(p2 - p1, p3 - p1));

		Vec3D n1 = normal;
		Vec3D n2 = normal;
		Vec3D n3 = normal;


		if (obj_triangle.v1.normal_index != -1)
		{
			n1 = vertex_normals[obj_triangle.v1.normal_index];
		}
		else if (smooth_normals)
		{
			n1 = computed_normals[obj_triangle.v1.position_index];
		}

		if (obj_triangle.v2.normal_index != -1)
		{
			n2 = vertex_normals[obj_triangle.v2.normal_index];
		}
		else if (smooth_normals)
		{
			n2 = computed_normals[obj_triangle.v2.position_index];
		}

		if (obj_triangle.v3.normal_index != -1)
		{
			n3 = vertex_normals[obj_triangle.v3.normal_index];
		}
		else if (smooth_normals)
		{
			n3 = computed_normals[obj_triangle.v3.position_index];
		}


		Vec2D uv1 = ZERO_VEC2D;
		Vec2D uv2 = ZERO_VEC2D;
		Vec2D uv3 = ZERO_VEC2D;

		if (obj_triangle.v1.texture_coords_index != -1)
		{
			uv1 = vertex_texture_coords[obj_triangle.v1.texture_coords_index];
		}

		if (obj_triangle.v2.texture_coords_index != -1)
		{
			uv2 = vertex_texture_coords[obj_triangle.v2.texture_coords_index];
		}

		if (obj_triangle.v3.texture_coords_index != -1)
		{
			uv3 = vertex_texture_coords[obj_triangle.v3.texture_coords_index];
		}


		Material material = materials[obj_triangle.material_name];

		triangles[i + prior_size] = { p1, p2, p3, n1, n2, n3, uv1, uv2, uv3, material };
	}


	// transform mesh position

	Bounding_Box bounding_box = get_bounding_box(prior_size, triangles);

	Vec3D center = { (bounding_box.min.x + bounding_box.max.x) * 0.5f, (bounding_box.min.y + bounding_box.max.y) * 0.5f, (bounding_box.min.z + bounding_box.max.z) * 0.5f };

	for (int i = prior_size; i < triangles.size(); ++i)
	{
		Triangle& triangle = triangles[i];

		triangle.p1 -= center;
		triangle.p2 -= center;
		triangle.p3 -= center;

		triangle.p1 = transformation.matrix * triangle.p1 + transformation.offset;
		triangle.p2 = transformation.matrix * triangle.p2 + transformation.offset;
		triangle.p3 = transformation.matrix * triangle.p3 + transformation.offset;

		triangle.n1 = normalize(transformation.matrix * triangle.n1);
		triangle.n2 = normalize(transformation.matrix * triangle.n2);
		triangle.n3 = normalize(transformation.matrix * triangle.n3);
	}
}