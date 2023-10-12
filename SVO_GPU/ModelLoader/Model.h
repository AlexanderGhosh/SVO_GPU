#pragma once
#include <glm.hpp>
#include <list>
#include "../Material.cuh"
#include "../3D/Octree3D.cuh"

class Model {
private:
	std::vector<node_t> trees_;
	std::vector<float3> treePositions_;
	std::vector<uint32_t> treeSizes_;
	Material materials_[MATERIAL_COUNT];
	uint32_t total_size_;
public:
	Model();
	Model(const std::list<tree_t>& trees, const std::list<float3>& positions);
	~Model();
	const node_t* getData() const;
	const uint32_t getTotalSize() const;
	const std::vector<uint32_t> getTreeSizes() const;
	const std::vector<float3> getTreePositions() const;
	const Material getMaterial(const uint8_t index) const;
};