#include "QB_ModelLoader.h"
#include <fstream>
#include "Model.h"
#include <iostream>

QB_Loader::QB_Loader() : ModelLoader(), max_span(0) {

}

Model QB_Loader::load(const std::string& file) {
	std::ifstream input(file, std::ios::binary | std::ios::in);
	uint8_t ch = input.get();
	buffer_t buffer;

	while (input.good()) {
		buffer.push_back(ch);
		ch = input.get();
	}


	uint32_t index = 0;
	uint32_t version = read_4(index, buffer);
	uint32_t colour_format = read_4(index, buffer);
	uint32_t z_axis = read_4(index, buffer);
	uint32_t compressed = read_4(index, buffer);
	uint32_t visibility_mask = read_4(index, buffer);
	uint32_t matrix_count = read_4(index, buffer);

	if (matrix_count > 1) {
		std::cout << "mat count larger than 1" << std::endl;
	}

	std::list<node_t> res;
	for (uint32_t i = 0; i < matrix_count; i++) {
		char mat_name_length = read_1(index, buffer);
		std::string mat_name = read_n(index, mat_name_length, buffer);

		uint32_t sizeX = read_4(index, buffer);
		uint32_t sizeY = read_4(index, buffer);
		uint32_t sizeZ = read_4(index, buffer);
		max_span = glm::ivec3(sizeX, sizeY, sizeZ);

		uint32_t posX = read_4(index, buffer);
		uint32_t posY = read_4(index, buffer);
		uint32_t posZ = read_4(index, buffer);




		// asumes that max_span is a square
		uchar4* ptr = (uchar4*)&buffer[index];
		inplace_vector data(ptr, sizeX * sizeY * sizeZ);
		std::vector<inplace_vector> modelData = {
			data
		};
		while (max_span.x > 8) {
			auto cpy = modelData;
			modelData.clear();
			for (auto& c : cpy) {
				auto split_ = splitData(c, max_span);
				for (auto& s : split_) {
					modelData.push_back(s);
				}
			}
			max_span /= 2;
		}

		std::list<_3D::Octree3D> all_nodes;
		//std::map<_3D::Octree3D*, glm::uvec3> all_parents_1;

		std::vector<tree_t> models;
		models.reserve(modelData.size());

		for (auto& data : modelData) {
			_3D::Octree3D root;
			std::list<_3D::Octree3D> nodes = {
				root
			};
			std::map<uchar3, uint32_t> colours;
			recursivlyMakeTree(data, root, { sizeX, sizeY, sizeZ }, nodes, colours);
			tree_t compiled = _3D::Octree3D::compile(&root);
			models.push_back(compiled);
		}
		return Model(models[0]);


		// std::list<_3D::Octree3D> all_nodes;
		// //std::map<_3D::Octree3D*, glm::uvec3> all_parents_1;

		// auto addNode = [&]() -> _3D::Octree3D& {
		// 	 all_nodes.push_back({});
		// 	 return all_nodes.back();
		// };
		// uchar4* ptr = (uchar4*) & buffer[index];
		// inplace_vector data(ptr, sizeX * sizeY * sizeZ);
		// _3D::Octree3D root;
		// std::list<_3D::Octree3D> nodes = {
		// 	 root
		// };
		// recursivlyMakeTree(data, root, { sizeX, sizeY, sizeZ }, nodes);
		// tree_t compiled = _3D::Octree3D::compile(&root);
		// return Model(compiled);
	}

}


// the parents span
std::array<QB_Loader::inplace_vector, 8> QB_Loader::splitData(const QB_Loader::inplace_vector& data, const glm::ivec3& span) const {
	std::array<inplace_vector, 8> res;
	glm::ivec3 half = span / 2;
	const uint32_t offset = max_span.x * max_span.y * half.z;
	const size_t size_ = half.x * half.y * half.z;

	res[0] = inplace_vector(data.ptr, size_);
	res[1] = inplace_vector(res[0].ptr + half.x, size_);
	res[2] = inplace_vector(res[0].ptr + max_span.x * half.y, size_);
	res[3] = inplace_vector(res[2].ptr + half.x, size_);

	res[4] = inplace_vector(res[0].ptr + offset, size_);
	res[5] = inplace_vector(res[1].ptr + offset, size_);
	res[6] = inplace_vector(res[2].ptr + offset, size_);
	res[7] = inplace_vector(res[3].ptr + offset, size_);
	return res;
}

void QB_Loader::recursivlyMakeTree(const QB_Loader::inplace_vector& data, _3D::Octree3D& parent, const glm::ivec3& span, std::list<_3D::Octree3D>& out, std::map<uchar3, uint32_t>& colours) const {
	auto addNode = [&]() -> _3D::Octree3D& {
		out.push_back({});
		return out.back();
		};
	auto split_data = splitData(data, span);
	if (data.size == 8) {
		for (int i = 0; i < 8; i++) {
			auto& d = split_data[i];
			if (d.ptr->w == 0) {
				continue;
			}
			parent.setChild(&addNode(), i);
		}
		return;
	}
	for (int i = 0; i < 8; i++) {
		auto& d = split_data[i];


		auto& child = addNode();
		recursivlyMakeTree(d, child, span / 2, out, colours);
		if (child.size() == 0) {
			out.pop_back();
			continue;
		}
		parent.setChild(&child, i);
	}
}
