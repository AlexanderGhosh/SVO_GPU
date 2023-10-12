#include "Model.h"

Model::Model() : trees_(), materials_(), total_size_(0), treePositions_(), treeSizes_() { }

Model::Model(const std::list<tree_t>& trees, const std::list<float3>& positions) : trees_(), treePositions_(positions.begin(), positions.end()), materials_(), total_size_(0), treeSizes_() {
	treeSizes_.reserve(trees.size());
	for (const auto& t : trees) {
		total_size_ += t.size();
		treeSizes_.push_back(t.size());
		trees_.insert(trees_.begin(), t.begin(), t.end());
	}
}

Model::~Model()
{
}

const node_t* Model::getData() const
{
	return trees_.data();
}

const uint32_t Model::getTotalSize() const
{
	return total_size_;
}

const std::vector<uint32_t> Model::getTreeSizes() const
{
	return treeSizes_;
}

const std::vector<float3> Model::getTreePositions() const
{
	return treePositions_;
}

const Material Model::getMaterial(const uint8_t index) const
{
	return materials_[index];
}
