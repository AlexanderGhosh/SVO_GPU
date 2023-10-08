#include <string>
#include <fstream>
#include <vector>
#include <iterator>

#include <glm.hpp>
#include <3D/Octree3D.cuh>

using uchar4 = glm::vec<4, uint8_t>;
union char4_uint32
{
	char c[4];
	uint32_t i;
	char4_uint32() : i(0) { }
	char4_uint32(char* d) : char4_uint32() {
		memcpy(c, d, 4);
	}
	char4_uint32(uint32_t i) : i(i) { }
};

uint32_t read_4(uint32_t& index, std::vector<char>& data) {
	char d[4] = {
		data[index],
		data[index + 1],
		data[index + 2],
		data[index + 3],
	};
	index += 4;
	return char4_uint32(d).i;
}

std::string read_n(uint32_t& index, char ammount, std::vector<char>& buffer) {
	std::string val;
	for (char i = 0; i < ammount; i++) {
		val += buffer[index++];
	}
	return val;
}

uint8_t read_1(uint32_t& index, std::vector<char>& data) {
	return char4_uint32(&data[index++]).i;
}


int main() {
	std::string fileLocation = "C:\\Users\\AGWDW\\Desktop\\untitled.qb";

	std::ifstream input(fileLocation, std::ios::binary);
	std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});

	uint32_t index = 0;
	uint32_t version = read_4(index, buffer);
	uint32_t colour_format = read_4(index, buffer);
	uint32_t z_axis = read_4(index, buffer);
	uint32_t compressed = read_4(index, buffer);
	uint32_t visibility_mask = read_4(index, buffer);
	uint32_t matrix_count = read_4(index, buffer);

	for (uint32_t i = 0; i < matrix_count; i++) {
		char mat_name_length = read_1(index, buffer);
		std::string mat_name = read_n(index, mat_name_length, buffer);

		uint32_t sizeX = read_4(index, buffer);
		uint32_t sizeY = read_4(index, buffer);
		uint32_t sizeZ = read_4(index, buffer);

		uint32_t posX = read_4(index, buffer);
		uint32_t posY = read_4(index, buffer);
		uint32_t posZ = read_4(index, buffer);

		if (!compressed) // if uncompressd
		{
			for (uint32_t y = 0; y < sizeY; y += 2) {
				for (uint32_t z = 0; z < sizeZ; z += 2) {
					for (uint32_t x = 0; x < sizeX; x += 2) {
						uint8_t idx = x + y * sizeX + z * sizeX * sizeY;
						_3D::Octree3D parent;
					}
				}
			}
		}

		int j = 0;
	}
	
	return 0;
}