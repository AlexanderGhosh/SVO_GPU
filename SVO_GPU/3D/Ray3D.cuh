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
		__device__ Ray3D();
		__device__ Ray3D(const float3& pos, const float3& dir);

		__device__ const float3 getPos() const;
		__device__ const float3 getDirection() const;
		__device__ const float3 t(const float3& a) const;

		__device__ const float3 point(const float t) const;
	};
}

