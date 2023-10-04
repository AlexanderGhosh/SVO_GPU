#pragma once
#include "cuda_runtime.h"
#include <math.h>

__device__ static float min(float3 a) {
	return fminf(a.x, fminf(a.y, a.z));
}

__device__ static float max(float3 a) {
	return fmaxf(a.x, fmaxf(a.y, a.z));
}

__device__ __host__ static float3 normalize(float3 a) {
	float sum = a.x * a.x + a.y * a.y + a.z * a.z;
	sum = sqrtf(sum);
	a.x /= sum;
	a.y /= sum;
	a.z /= sum;
	return a;
}

__device__ static float3 sign(float3 a) {
	auto s = [](float b) -> float {
		if (b > 0) return 1;
		if (b < 0) return -1;
		else return 0;
		};
	return make_float3(s(a.x), s(a.y), s(a.z));
}

__device__ static float3 abs(float3 a) {
	return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

__device__ static float3 operator+ (float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static float3 operator- (float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ static float3 operator* (float3 a, float3 b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ static float3 operator/ (float3 a, float3 b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ static float3 operator+ (float3 a, float b) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ static float3 operator* (float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}



__device__ static void operator-= (float3& a, float3 b) {
	a = a - b;
}
__device__ static void operator-= (float3& a, float b) {
	a = make_float3(a.x - b, a.y - b, a.z - b);
}

__device__ static void operator*= (float3& a, float b) {
	a = make_float3(a.x * b, a.y * b, a.z * b);
}