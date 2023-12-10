#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <iostream>

#include "macros.h"
#include "math_library.cuh"
#include "scene.cuh"


struct Camera
{
	Vec3D position;
	float yaw, pitch;
	float FOV;

	__host__ __device__ Matrix3X3 rotation()
	{
		return rotation_matrix(yaw, pitch);
	}
};

inline void camera_movement(GLFWwindow* window, Camera& camera, float time_step, int& sample_count)
{
	float movement_speed = 0.5 * time_step;

	Vec3D motion_vector = ZERO_VEC3D;

	if (glfwGetKey(window, GLFW_KEY_W))
	{
		motion_vector = camera.rotation() * Vec3D { 0.0f, 0.0f, 1.0f } * movement_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_A))
	{
		motion_vector = camera.rotation() * Vec3D{ -1.0f, 0.0f, 0.0f } * movement_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_S))
	{
		motion_vector = camera.rotation() * Vec3D{ 0.0f, 0.0f, -1.0f } * movement_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_D))
	{
		motion_vector = camera.rotation() * Vec3D{ 1.0f, 0.0f, 0.0f } * movement_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE))
	{
		motion_vector = Vec3D{ 0.0f, 1.0f, 0.0f } * movement_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT))
	{
		motion_vector = Vec3D{ 0.0f, -1.0f, 0.0f } * movement_speed;

		sample_count = 1;
	}

	camera.position += motion_vector;


	float rotation_speed = 2 * time_step;

	if (glfwGetKey(window, GLFW_KEY_LEFT))
	{
		camera.yaw -= rotation_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_RIGHT))
	{
		camera.yaw += rotation_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_UP))
	{
		camera.pitch -= rotation_speed;

		sample_count = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN))
	{
		camera.pitch += rotation_speed;

		sample_count = 1;
	}
}