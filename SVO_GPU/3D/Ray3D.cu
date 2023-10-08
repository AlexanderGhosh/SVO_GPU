#include "Ray3D.cuh"
#include "../Constants.cuh"
#include "../VectorMaths.cuh"

using namespace _3D;
Ray3D::Ray3D() : pos_(), dir_(), m_coef(), c_coef()
{
}

Ray3D::Ray3D(const float3& pos, const float3& dir) : Ray3D()
{
	pos_ = pos;
	dir_ = normalize(dir);

	if (fabsf(dir_.x) < EPSILON) dir_.x = copysignf(EPSILON, dir_.x);
	if (fabsf(dir_.y) < EPSILON) dir_.y = copysignf(EPSILON, dir_.y);
	if (fabsf(dir_.z) < EPSILON) dir_.z = copysignf(EPSILON, dir_.z);

	m_coef = make_float3(1, 1, 1) / dir_;
	c_coef = m_coef * pos_;
	
}

const float3 Ray3D::getPos() const
{
	return pos_;
}

const float3 Ray3D::getDirection() const
{
	return dir_;
}

const float3 Ray3D::t(const float3& a) const
{
	return (m_coef * a) - c_coef;
}

__device__ const float3 _3D::Ray3D::point(const float t) const
{
	return pos_ + dir_ * t;
}
