#pragma once
#include "cuda_runtime.h"
#include <glm.hpp>

constexpr float EPSILON = 0.001f;
constexpr int MAX_SCALE = 8;
constexpr int MIN_SCALE = 1;
constexpr int MAX_ITTERATIONS = 1000;
constexpr int OCTREE_SIZE = 8;
constexpr int PARENT_STACK_DEPTH = 3;

constexpr int X_RESOLUTION = 640;
constexpr int Y_RESOLUTION = 640;
constexpr int PIXEL_COUNT = X_RESOLUTION * Y_RESOLUTION;
constexpr auto NUM_CHANNELS = 3;
constexpr int  IMAGE_DATA_SIZE = PIXEL_COUNT * NUM_CHANNELS;

struct node_t {
	unsigned int child_data;
	unsigned int shader_data;
};

using shader_t = uchar4;
using slot_t = int; // at the moment it CAN'T be unsigned
using mirror_t = float3; // glm::bvec2 shold migrate to this
// using node_t = unsigned int;

__device__ __host__ static uchar4* element(uchar4* arr, size_t pitch, int x, int y) {
	return (uchar4*)((char*)arr + y * pitch) + x;
}

__device__ __host__ static int index2D(int x, int y) {
	return (X_RESOLUTION * y) + x;
}

__device__ __host__ static int index3D(int x, int y, int z) {
	return (z * X_RESOLUTION * Y_RESOLUTION) + (y * X_RESOLUTION) + x;
}

__device__ __host__ static glm::vec3 make_vec3(const float3& a) {
	return { a.x, a.y, a.z };
}

__device__ __host__ static float3 make_float3(const glm::vec3& a) {
	return { a.x, a.y, a.z };
}

__device__ __host__ static glm::mat3 get_rotation(const glm::vec3& a, const glm::vec3& b) {
	auto v = glm::cross(a, b);
	auto c = glm::dot(a, b);
	
	auto sk = glm::mat3(0);
	sk =
	{
		{ 0, -v.z, v.y },
		{ v.z, 0, -v.x },
		{ -v.y, v.x, 0 }
	};

	auto t = 1.f / (1 + c);

	auto res = glm::mat3(1) + sk + (sk * sk) * t;
	return res;
}



#define IS_3D