#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "macros.h"
#include "scene.cuh"
#include "camera.cuh"
#include "screen.cuh"
#include "math_library.cuh"


struct Sample
{
    Material material;
    Vec3D position;
    Vec3D normal, tangent, bitangent;
};

__device__ Vec3D sample_sky(Vec3D ray_direction, Scene scene, float multiplier)
{
    Vec2D sample;

    sample.x = (sign(ray_direction.z) * acosf(ray_direction.x / sqrtf(ray_direction.x * ray_direction.x + ray_direction.z * ray_direction.z)) + PI) / TAU;
    sample.y = acosf(ray_direction.y) / PI;


    int pixel_number = int(sample.y * scene.sky_texture.height) * scene.sky_texture.width + (sample.x * scene.sky_texture.width);

    uchar4 color = scene.sky_texture.buffer[pixel_number];

    return Vec3D{ color.x / 255.0f, color.y / 255.0f, color.z / 255.0f } * multiplier;
}

__device__ Vec3D calculate_barycentric_coordinates(Vec3D point_on_plane, Triangle triangle) // uses Cramer's rule to solve for barycentric coordinates, source: https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
{
    Vec3D v0 = triangle.p2 - triangle.p1;
    Vec3D v1 = triangle.p3 - triangle.p1;
    Vec3D v2 = point_on_plane - triangle.p1;

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);

    float reciprocal_denominator = 1.0f / (d00 * d11 - d01 * d01);


    Vec3D barycentric_coordinates = ZERO_VEC3D;

    barycentric_coordinates.y = (d11 * d20 - d01 * d21) * reciprocal_denominator;
    barycentric_coordinates.z = (d00 * d21 - d01 * d20) * reciprocal_denominator;
    barycentric_coordinates.x = 1.0f - barycentric_coordinates.y - barycentric_coordinates.z;


    return barycentric_coordinates;
}

__device__ bool intersect_triangle(Ray ray, Triangle triangle, Vec3D* barycentric_coordinates, float* t)
{
    float direction_dot_normal = dot(ray.direction, triangle.normal);


    if (direction_dot_normal == 0) // ray is parallell to triangle
    {
       return false;
    }


    float d = dot(triangle.normal, triangle.p1);


    float s = (d - dot(ray.position, triangle.normal)) / direction_dot_normal;

    if (s < 0.0001f) // no intersection
    {
        return false;
    }

    *t = s;


    Vec3D point_on_plane = ray.position + s * ray.direction;

    Vec3D barycentric = calculate_barycentric_coordinates(point_on_plane, triangle);


    if (barycentric.x >= 0.0f && barycentric.x <= 1.0f && barycentric.y >= 0.0f && barycentric.y <= 1.0f && barycentric.z >= 0.0f && barycentric.z <= 1.0f) // point is inside triangle
    {
        *barycentric_coordinates = barycentric;

        return true;
    }

    return false;
}

__device__ bool trace_ray(Ray ray, Scene scene, Sample& sample)
{
	float smallest_t = FLT_MAX;
	bool hit = false;


	for (int i = 0; i < scene.triangle_count; ++i)
	{
		Triangle triangle = scene.triangles[i];

		
        Vec3D barycentric_coordinates = ZERO_VEC3D;
        float t = FLT_MAX;

        if (intersect_triangle(ray, triangle, &barycentric_coordinates, &t) && (t < smallest_t))
        {
            hit = true;

            sample.material = triangle.material;
            sample.position = barycentric_coordinates.x * triangle.p1 + barycentric_coordinates.y * triangle.p2 + barycentric_coordinates.z * triangle.p3;
            sample.normal = triangle.normal;
            sample.tangent = triangle.tangent;
            sample.bitangent = triangle.bitangent;

            smallest_t = t;
        }
	}


    return hit;
}