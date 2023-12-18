#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <math.h>

#define PI 3.1415926536f
#define TAU (PI * 2)
#define HALF_PI (PI / 2)

#define ZERO_VEC2D { 1.0f, 1.0f }
#define ZERO_VEC3D { 0.0f, 0.0f, 0.0f }


__host__ __device__ float square(float x)
{
	return x * x;
}

__host__ __device__ float clamp(float x, float lower, float upper)
{
	return fmaxf(lower, fminf(upper, x));
}

__host__ __device__ float sign(float x)
{
	return (x >= 0) ? 1.0f : -1.0f;
}

__device__ float gamma_correction(float x)
{
	float output = 12.92 * x;

	if (x > 0.0031308)
	{
		output = 1.055 * powf(x, 1.0 / 2.4) - 0.055;
	}

	return output;
}


struct Vec2D
{
	union
	{
		float x, u;
	};

	union
	{
		float y, v;
	};
};

__host__ __device__ Vec2D operator + (Vec2D v1, Vec2D v2)
{
	return { v1.x + v2.x, v1.y + v2.y };
}

__host__ __device__ Vec2D operator - (Vec2D v1, Vec2D v2)
{
	return { v1.x - v2.x, v1.y - v2.y };
}

__host__ __device__ Vec2D operator * (Vec2D v, float s)
{
	return { v.x * s, v.y * s };
}

__host__ __device__ Vec2D operator * (float s, Vec2D v)
{
	return { v.x * s, v.y * s };
}

__host__ __device__ Vec2D operator * (Vec2D v1, Vec2D v2)
{
	return { v1.x * v2.x, v1.y * v2.y };
}

__host__ __device__ float dot(Vec2D v1, Vec2D v2)
{
	return v1.x * v2.x + v1.y * v2.y;
}


struct Vec3D
{
	union
	{
		float x, r;
	};

	union
	{
		float y, g;
	};

	union
	{
		float z, b;
	};
};

__host__ __device__ Vec3D operator + (Vec3D v1, Vec3D v2)
{
	return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

__host__ __device__ Vec3D operator - (Vec3D v1, Vec3D v2)
{
	return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

__host__ __device__ Vec3D operator - (Vec3D v)
{
	return { -v.x, -v.y, -v.z };
}

__host__ __device__ Vec3D operator * (Vec3D v, float s)
{
	return { v.x * s, v.y * s, v.z * s };
}

__host__ __device__ Vec3D operator * (float s, Vec3D v)
{
	return { v.x * s, v.y * s, v.z * s };
}

__host__ __device__ Vec3D operator * (Vec3D v1, Vec3D v2)
{
	return { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
}

__host__ __device__ void operator += (Vec3D& v1, Vec3D v2)
{
	v1.x += v2.x;
	v1.y += v2.y;
	v1.z += v2.z;
}

__host__ __device__ void operator -= (Vec3D& v1, Vec3D v2)
{
	v1.x -= v2.x;
	v1.y -= v2.y;
	v1.z -= v2.z;
}

__host__ __device__ void operator *= (Vec3D& v1, Vec3D v2)
{
	v1.x *= v2.x;
	v1.y *= v2.y;
	v1.z *= v2.z;
}

__host__ __device__ void operator *= (Vec3D& v1, float s)
{
	v1.x *= s;
	v1.y *= s;
	v1.z *= s;
}

__host__ __device__ bool operator == (Vec3D v1, Vec3D v2)
{
	return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
}

__host__ __device__ bool operator != (Vec3D v1, Vec3D v2)
{
	return !(v1 == v2);
}

__host__ __device__ uint4 operator + (uint4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uchar4 a, uchar4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uint4 a, uchar4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uchar4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ void scale(Vec3D* v, float s)
{
	v->x *= s;
	v->y *= s;
	v->z *= s;
}

__host__ __device__ float dot(Vec3D v1, Vec3D v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ Vec3D cross(Vec3D v1, Vec3D v2)
{
	return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
}

__host__ __device__ float magnitude(Vec3D v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ float magnitude_squared(Vec3D v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ Vec3D normalize(Vec3D v)
{
	float reciprocal_magnitude = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	return { v.x * reciprocal_magnitude, v.y * reciprocal_magnitude, v.z * reciprocal_magnitude };
}

__host__ __device__ void normalize(Vec3D* v)
{
	float reciprocal_magnitude = 1.0f / sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);

	v->x *= reciprocal_magnitude;
	v->y *= reciprocal_magnitude;
	v->z *= reciprocal_magnitude;
}

__host__ __device__ Vec3D lerp(Vec3D v1, Vec3D v2, float t)
{
	return v1 + t * (v2 - v1);
}

__device__ Vec3D proj(Vec3D v1, Vec3D v2)
{
	return (dot(v1, v2) / magnitude_squared(v2)) * v2;
}

__device__ Vec3D gamma_correction(Vec3D color)
{
	return { gamma_correction(color.r), gamma_correction(color.g), gamma_correction(color.b) };
}

__device__ Vec3D tone_mapping(Vec3D color)
{
	return color;
}


struct Vec4D
{
	union
	{
		float x, r;
	};

	union
	{
		float y, g;
	};

	union
	{
		float z, b;
	};

	union
	{
		float w, a;
	};
};


struct Int3
{
	union
	{
		int x, r;
	};

	union
	{
		int y, g;
	};

	union
	{
		int z, b;
	};
};


struct Ray
{
	Vec3D position;
	Vec3D direction;
};


struct Matrix3X3
{
	Vec3D i_hat;
	Vec3D j_hat;
	Vec3D k_hat;
};

__host__ __device__ Matrix3X3 operator * (Matrix3X3 m, float s)
{
	return { s * m.i_hat, s * m.j_hat, s * m.k_hat };
}

__host__ __device__ Vec3D operator * (Matrix3X3 m, Vec3D v)
{
	return v.x * m.i_hat + v.y * m.j_hat + v.z * m.k_hat;
}

__host__ __device__ Matrix3X3 operator * (Matrix3X3 m2, Matrix3X3 m1)
{
	return { m2 * m1.i_hat, m2 * m1.j_hat, m2 * m1.k_hat };
}

__host__ __device__ Matrix3X3 rotation_matrix(float yaw, float pitch = 0.0f, float roll = 0.0f)
{
	Matrix3X3 y_axis_rotation =
	{
		{ cos(yaw), 0.0f, -sin(yaw) },
		{ 0.0f, 1.0f, 0.0f },
		{ sin(yaw), 0.0f, cos(yaw) }
	};

	Matrix3X3 x_axis_rotation =
	{
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, cos(pitch), sin(pitch) },
		{ 0.0f, -sin(pitch), cos(pitch) }
	};

	Matrix3X3 z_axis_rotation =
	{
		{ cos(roll), sin(roll), 0.0f },
		{ -sin(roll), cos(roll), 0.0f },
		{ 0.0f, 0.0f, 1.0f }
	};

	return z_axis_rotation * y_axis_rotation * x_axis_rotation;
}

__host__ __device__ Matrix3X3 scale_matrix(float scale)
{
	Matrix3X3 matrix =
	{
		{ scale, 0.0f, 0.0f },
		{ 0.0f, scale, 0.0f },
		{ 0.0f, 0.0f, scale }
	};

	return matrix;
}