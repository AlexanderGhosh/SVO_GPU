#pragma once
#include <glm.hpp>
#include "../3D/Octree3D.cuh"

class Model {
private:
	tree_t tree_;
	material_t materials_[MATERIAL_COUNT];
	uint32_t tree_size_;
public:
	Model();
	Model(tree_t compiled);
	~Model();
	const node_t* getData() const;
	const uint32_t getSize() const;
	const material_t getMaterial(const uint8_t index) const;
};