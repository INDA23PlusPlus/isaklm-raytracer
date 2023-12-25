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
    Vec3D albedo;
    Vec3D emittance;
    float roughness;
    float refractive_index;
    float extinction;
    bool transparent;

    Vec3D position;
    Vec3D normal, tangent, bitangent;
};

__device__ Vec3D sample_texture(Texture texture, Vec3D color_blend, Vec2D texture_coordinates)
{
    if (texture.buffer == nullptr)
    {
        return color_blend;
    }


    int pixel_number = int(texture_coordinates.y * texture.height) * texture.width + (texture_coordinates.x * texture.width);

    uchar4 color = texture.buffer[pixel_number];

    return Vec3D{ color.x / 255.0f, color.y / 255.0f, color.z / 255.0f } * color_blend;
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
    Vec3D normal = normalize(cross(triangle.p2 - triangle.p1, triangle.p3 - triangle.p1));


    float direction_dot_normal = dot(ray.direction, normal);


    if (direction_dot_normal == 0) // ray is parallell to triangle
    {
       return false;
    }


    float d = dot(normal, triangle.p1);


    float s = (d - dot(ray.position, normal)) / direction_dot_normal;

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

__device__ bool trace_leaf_node(Ray ray, float max_t, int index_offset, int triangle_count, int* triangle_indicies, Triangle* triangles, Sample& sample)
{
    float smallest_t = max_t;
    bool hit = false;


    for (int i = 0; i < triangle_count; ++i)
    {
        Triangle triangle = triangles[triangle_indicies[index_offset + i]];


        Vec3D barycentric_coordinates = ZERO_VEC3D;
        float t = FLT_MAX;

        if (intersect_triangle(ray, triangle, &barycentric_coordinates, &t) && (t < smallest_t))
        {
            hit = true;
            smallest_t = t;

            Vec2D texture_coordinates = triangle.t1 * barycentric_coordinates.x + triangle.t2 * barycentric_coordinates.y + triangle.t3 * barycentric_coordinates.z;

            sample.albedo = sample_texture(triangle.material.texture, triangle.material.albedo, texture_coordinates);
            sample.emittance = sample_texture(triangle.material.texture, triangle.material.emittance, texture_coordinates);
            sample.roughness = triangle.material.roughness;
            sample.refractive_index = triangle.material.refractive_index;
            sample.extinction = triangle.material.extinction;
            sample.transparent = triangle.material.transparent;

            sample.position = barycentric_coordinates.x * triangle.p1 + barycentric_coordinates.y * triangle.p2 + barycentric_coordinates.z * triangle.p3;

            sample.normal = normalize(barycentric_coordinates.x * triangle.n1 + barycentric_coordinates.y * triangle.n2 + barycentric_coordinates.z * triangle.n3);
            sample.tangent = normalize(cross(sample.normal, triangle.p2 - triangle.p1));
            sample.bitangent = cross(sample.normal, sample.tangent);

            if (dot(ray.direction, sample.normal) > 0) // flip the normal if it is pointing in the wrong direction
            {
                sample.normal = -sample.normal;
            }
        }
    }


    return hit;
}

__device__ bool ray_behind_plane(Ray ray, int plane_axis, float plane_offset)
{
    if (plane_axis == 0)
    {
        return (ray.position.x >= plane_offset);
    }
    else if (plane_axis == 1)
    {
        return (ray.position.y >= plane_offset);
    }
    else
    {
        return (ray.position.z >= plane_offset);
    }
}

__device__ float intersect_plane(Ray ray, int plane_axis, float plane_offset)
{
    if (plane_axis == 0)
    {
        float distance = plane_offset - ray.position.x;

        return distance / ray.direction.x;
    }
    else if (plane_axis == 1)
    {
        float distance = plane_offset - ray.position.y;

        return distance / ray.direction.y;
    }
    else
    {
        float distance = plane_offset - ray.position.z;

        return distance / ray.direction.z;
    }
}

__device__ bool intersect_bounding_box(Ray ray, Bounding_Box bounding_box, float& t1, float& t2) // slab method https://tavianator.com/2011/ray_box.html
{
    Vec3D t_min;
    t_min.x = (bounding_box.min.x - ray.position.x) / ray.direction.x;
    t_min.y = (bounding_box.min.y - ray.position.y) / ray.direction.y;
    t_min.z = (bounding_box.min.z - ray.position.z) / ray.direction.z;

    Vec3D t_max;
    t_max.x = (bounding_box.max.x - ray.position.x) / ray.direction.x;
    t_max.y = (bounding_box.max.y - ray.position.y) / ray.direction.y;
    t_max.z = (bounding_box.max.z - ray.position.z) / ray.direction.z;

    Vec3D s1;
    s1.x = fminf(t_min.x, t_max.x);
    s1.y = fminf(t_min.y, t_max.y);
    s1.z = fminf(t_min.z, t_max.z);

    Vec3D s2;
    s2.x = fmaxf(t_min.x, t_max.x);
    s2.y = fmaxf(t_min.y, t_max.y);
    s2.z = fmaxf(t_min.z, t_max.z);

    float t_near = fmaxf(fmaxf(s1.x, s1.y), s1.z);
    float t_far = fminf(fminf(s2.x, s2.y), s2.z);

    t1 = t_near;
    t2 = t_far;


    return (t_near <= t_far);
}

__device__ bool trace_ray(Ray ray, Scene scene, Sample& sample)
{
    int node_indicies[KD_TREE_DEPTH];
    float entry_distances[KD_TREE_DEPTH];
    float exit_distances[KD_TREE_DEPTH];

    float t1, t2;

    if (!intersect_bounding_box(ray, scene.kd_tree.bounding_box, t1, t2))
    {
        return false;
    }

    node_indicies[0] = 0;
    entry_distances[0] = t1;
    exit_distances[0] = t2;

    int stack_index = 1;


    while (stack_index > 0)
    {
        --stack_index; // pop stack
        KD_Tree_Node node = scene.kd_tree.nodes[node_indicies[stack_index]];

        float entry_distance = entry_distances[stack_index];
        float exit_distance = exit_distances[stack_index];


        while (!node.is_leaf_node)
        {
            int near_child_index = node.child_index1;
            int far_child_index = node.child_index2;

            if (ray_behind_plane(ray, node.plane_axis, node.plane_offset))
            {
                near_child_index = node.child_index2;
                far_child_index = node.child_index1;
            }


            float t = intersect_plane(ray, node.plane_axis, node.plane_offset);


            if (t >= exit_distance || t < 0)
            {
                node = scene.kd_tree.nodes[near_child_index];
            }
            else if (t <= entry_distance)
            {
                node = scene.kd_tree.nodes[far_child_index];
            }
            else
            {
                node_indicies[stack_index] = far_child_index;
                entry_distances[stack_index] = t;
                exit_distances[stack_index] = exit_distance;
                ++stack_index;

                node = scene.kd_tree.nodes[near_child_index];
                exit_distance = t;
            }
        }
        
        if (node.triangle_count > 0)
        {
            if (trace_leaf_node(ray, exit_distance, node.index_offset, node.triangle_count, scene.kd_tree.triangle_indicies, scene.triangles, sample))
            {
                return true;
            }
        }
    }

    return false;
}