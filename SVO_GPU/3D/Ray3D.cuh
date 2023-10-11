#pragma once
#include "cuda_runtime.h"

namespace _3D {
	class Ray3D
	{
	private:
		float3 pos_;
		float3 dir_;
		float3 m_coef;
		float3 c_coef;
	public:
		float3 dir_inv_;
		__host__ __device__ Ray3D();
		__host__ __device__ Ray3D(const float3& pos, const float3& dir);

		__host__ __device__ const float3 getPos() const;
		__host__ __device__ const float3 getDirection() const;
		__host__ __device__ const float3 t(const float3& a) const;

		__host__ __device__ const float3 point(const float t) const;
	};
}

