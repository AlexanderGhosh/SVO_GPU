#include "Octree3D.cuh"

#include <list>

using namespace _3D;
Octree3D::Octree3D() : parent(nullptr), children()
{
}

void Octree3D::setChild(Octree3D* child, slot_t slot)
{
	child->parent = this;
	children[slot] = child;
}

const bool Octree3D::isValid(slot_t slot) const
{
	return children[slot];
}

const bool Octree3D::isLeaf() const
{
	return size() == 0;
}

const size_t Octree3D::size() const
{
	uint32_t counter = 0;
	for (const auto& child : children) {
		if (child) {
			counter++;
		}
	}
	return counter;
}

const node_t Octree3D::toInt32() const
{
	uint32_t valid = 0;
	uint32_t leaf = 0;
	
	for (auto itt = children.rbegin(); itt != children.rend(); itt++) {
		const Octree3D* child = *itt;
		valid <<= 1;
		leaf <<= 1;
		if (child) {
			valid++;
			if (child->isLeaf()) {
				leaf++;
			}
		}
	}
	valid <<= 8;
	return valid | leaf;
}

std::vector<node_t> Octree3D::compile(Octree3D* root)
{
	std::vector<uint32_t> result = {};


	// Mark all the vertices as not visited
	std::vector<Octree3D*> visited;

	// Create a queue for BFS
	std::list<Octree3D*> queue;

	// Mark the current node as visited and enqueue it
	visited.push_back(root);
	queue.push_back(root);

	while (!queue.empty()) {

		// Dequeue a vertex from queue and print it
		Octree3D* s = queue.front();
		queue.pop_front();

		// Get all child vertices of the dequeued
		// vertex s.
		// If an child has not been visited,
		// then mark it visited and enqueue it
		for (auto child : s->children) {
			if (child && !child->isLeaf()) {
				if (std::find(visited.begin(), visited.end(), child) == visited.end()) {
					// not found
					visited.push_back(child);
					queue.push_back(child);
				}
			}
		}
	}


	for (int i = 0; i < visited.size(); i++) {
		Octree3D* svo = visited[i];
		if (svo->parent) {
			auto itter = std::find(visited.begin(), visited.end(), svo->parent);
			if (itter != visited.end()) {
				int parent_index = itter - visited.begin();
				uint32_t& parent = result[parent_index];
				if (!(parent & 0xffff0000)) {
					uint32_t idx = i << 16;
					parent |= idx;
				}
			}
		}
		const uint32_t compiled = svo->toInt32();
		result.push_back(compiled);
	}

	return result;
}

std::vector<Octree3D> Octree3D::getDefault()
{

	//std::vector<Octree3D> result(10);

	//Octree3D& root = result[0];
	//Octree3D& c0 = result[1]; // leef
	//Octree3D& c1 = result[2];
	//Octree3D& c2 = result[3];
	//Octree3D& c3 = result[4];
	//Octree3D& c1_0 = result[5]; // leef
	//Octree3D& c2_2 = result[6];
	//Octree3D& c2_2_1 = result[7]; // leef
	//Octree3D& c3_1 = result[8];
	//Octree3D& c3_1_3 = result[9]; // leef

	//c1.setChild(&c1_0, 0);

	//c2.setChild(&c2_2, 2);
	//c2_2.setChild(&c2_2_1, 1);

	//c3.setChild(&c3_1, 1);
	//c3_1.setChild(&c3_1_3, 3);

	//root.setChild(&c0, 0);
	//root.setChild(&c1, 1);
	//root.setChild(&c2, 2);
	//root.setChild(&c3, 3);

	std::vector<Octree3D> result(25);

	Octree3D& root = result[0];
	Octree3D& c0 = result[1];
	Octree3D& c1 = result[2];
	Octree3D& c2 = result[3];
	Octree3D& c3 = result[4];
	Octree3D& c0_0 = result[5];
	Octree3D& c1_1 = result[6];
	Octree3D& c2_2 = result[7];
	Octree3D& c3_3 = result[8];
	Octree3D& c0_0_0 = result[9];
	Octree3D& c1_1_1 = result[10];
	Octree3D& c2_2_2 = result[11];
	Octree3D& c3_3_3 = result[12];

	Octree3D& c4 = result[13];
	Octree3D& c5 = result[14];
	Octree3D& c6 = result[15];
	Octree3D& c7 = result[16];
	Octree3D& c4_4 = result[17];
	Octree3D& c5_5 = result[18];
	Octree3D& c6_6 = result[19];
	Octree3D& c7_7 = result[20];
	Octree3D& c4_4_4 = result[21];
	Octree3D& c5_5_5 = result[22];
	Octree3D& c6_6_6 = result[23];
	Octree3D& c7_7_7 = result[24];


	c0.setChild(&c0_0, 0);
	c0_0.setChild(&c0_0_0, 0);

	c1.setChild(&c1_1, 1);
	c1_1.setChild(&c1_1_1, 1);

	c2.setChild(&c2_2, 2);
	c2_2.setChild(&c2_2_2, 2);

	c3.setChild(&c3_3, 3);
	c3_3.setChild(&c3_3_3, 3);

	c4.setChild(&c4_4, 4);
	c4_4.setChild(&c4_4_4, 4);

	c5.setChild(&c5_5, 5);
	c5_5.setChild(&c5_5_5, 5);

	c6.setChild(&c6_6, 6);
	c6_6.setChild(&c6_6_6, 6);

	c7.setChild(&c7_7, 7);
	c7_7.setChild(&c7_7_7, 7);

	root.setChild(&c0, 0);
	root.setChild(&c1, 1);
	root.setChild(&c2, 2);
	root.setChild(&c3, 3);
	root.setChild(&c4, 4);
	root.setChild(&c5, 5);
	root.setChild(&c6, 6);
	root.setChild(&c7, 7);

	return result;
}								   

