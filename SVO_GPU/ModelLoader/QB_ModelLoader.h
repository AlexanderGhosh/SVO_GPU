#pragma once
#include <glm.hpp>
#include <array>
#include <list>
#include <map>
#include "ModelLoader.h"
#include "../3D/Octree3D.cuh"

class Model;
class QB_Loader : public ModelLoader {
private:
	struct inplace_vector {
		size_t size;
		uchar4* ptr;
		inplace_vector() : ptr(nullptr), size(0) { }
		inplace_vector(uchar4* ptr, size_t size) : ptr(ptr), size(size) { }
		uchar4 operator[] (uint32_t t) {
			return *(ptr + t);
		}
	};

	glm::ivec3 max_span;
	std::array<inplace_vector, 8> splitData(const inplace_vector& data, const glm::ivec3& span) const;
	void recursivlyMakeTree(const inplace_vector& data, _3D::Octree3D& parent, const glm::ivec3& span, std::list<_3D::Octree3D>& out, std::map<uchar3, uint32_t>& colours) const;
public:
	QB_Loader();
	Model load(const std::string& file) override;
};
