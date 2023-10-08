#include "Model.h"

Model::Model() : tree_(), materials_(), tree_size_(0) { }

Model::Model(tree_t compiled) : tree_(compiled), materials_(), tree_size_(compiled.size()) {
	materials_[0] = make_uchar4(0, 0, 0, 255);

	/*tree_ = (node_t*)malloc(tree_size_);
	memcpy(tree_, compiled.data(), tree_size_);*/
}

Model::~Model()
{
}

const node_t* Model::getData() const
{
	return tree_.data();
}

const uint32_t Model::getSize() const
{
	return tree_size_;
}

const material_t Model::getMaterial(const uint8_t index) const
{
	return materials_[index];
}
