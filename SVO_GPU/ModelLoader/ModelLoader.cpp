#include "ModelLoader.h"

ModelLoader::uchar4_uint32::uchar4_uint32() : i_(0)  { }

ModelLoader::uchar4_uint32::uchar4_uint32(uint8_t* d) : uchar4_uint32() {
	memcpy(c_, d, 4);
}
ModelLoader::uchar4_uint32::uchar4_uint32(uint32_t i) : i_(i) { }

const uint8_t* ModelLoader::uchar4_uint32::getUChar4() const {
	return c_;
}

const uint32_t ModelLoader::uchar4_uint32::getUint32() const {
	return i_;
}

ModelLoader::ModelLoader() { }

uint8_t ModelLoader::read_1(uint32_t& index, buffer_t& buffer) const {
	return buffer[index++];
}

uint32_t ModelLoader::read_4(uint32_t& index, buffer_t& data) const {
	uint8_t d[4] = {
		data[index],
		data[index + 1],
		data[index + 2],
		data[index + 3],
	};
	index += 4;
	return uchar4_uint32(d).getUint32();
}

std::string ModelLoader::read_n(uint32_t& index, char ammount, buffer_t& buffer) const {
	std::string val;
	for (char i = 0; i < ammount; i++) {
		val += buffer[index++];
	}
	return val;
}