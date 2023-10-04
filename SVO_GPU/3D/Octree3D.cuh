#pragma once
#include <array>
#include <vector>
#include "../Constants.cuh"

namespace _3D {
	class Octree3D {
	private:
		std::array<Octree3D*, 8> children;
		Octree3D* parent;
	public:
		Octree3D();

		void setChild(Octree3D* child, slot_t slot);

		const bool isValid(slot_t slot) const;

		const bool isLeaf() const;

		const size_t size() const;

		const node_t toInt32() const;

		static std::vector<node_t> compile(Octree3D* root);

		static std::vector<Octree3D> getDefault();
	};
}
