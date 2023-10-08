#pragma once
#include <string>
#include <vector>

class Model;
class ModelLoader {
private:
	union uchar4_uint32
	{
	private:
		uint8_t c_[4];
		uint32_t i_;
	public:
		uchar4_uint32();
		uchar4_uint32(uint8_t* d);
		uchar4_uint32(uint32_t i);

		const uint8_t* getUChar4() const;
		const uint32_t getUint32() const;
	};
protected:
	using buffer_t = std::vector<uint8_t>;
	ModelLoader();
	uint8_t read_1(uint32_t& index, buffer_t& buffer) const;
	uint32_t read_4(uint32_t& index, buffer_t& buffer) const;
	std::string read_n(uint32_t& index, const char size, buffer_t& buffer) const;
public:
	virtual Model load(const std::string& model) = 0;
};