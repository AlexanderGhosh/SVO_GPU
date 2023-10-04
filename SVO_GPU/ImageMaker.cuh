#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <array>
#include "Constants.cuh"

// using imagedata_t = std::array<std::array<std::array<float, NUM_CHANNELS>, Y_RESOLUTION>, X_RESOLUTION>;
using imagedata_t = float3[PIXEL_COUNT];

void Default(const std::string& fileName, int xSize, int ySize) {
	std::ofstream file;
	file.open(fileName);
	file << "P3\n" << xSize << " " << ySize << "\n255\n";
	for (int j = ySize - 1; j >= 0; j--) {
		for (int i = 0; i < xSize; i++) {
			float r = float(i) / float(xSize);
			float g = float(j) / float(ySize);
			float b = .2f;
			int ir = 255.99f * r;
			int ig = 255.99f * g;
			int ib = 255.99f * b;
			file << ir << " " << ig << " " << b << "\n";
		}
	}
	file.close();
	std::cout << "Finished Writing to: " << fileName << std::endl;
}

void SaveImage(const std::string& fileName, int xSize, int ySize, float3* data) {
	std::ofstream file;
	file.open(fileName);
	file << "P3\n" << xSize << " " << ySize << "\n255\n";
	for (int j = ySize - 1; j >= 0; j--) {
		for (int i = 0; i < xSize; i++) {

			int idx = index2D(i, j);

			float& r = data[idx].x;
			float& g = data[idx].y;
			float& b = data[idx].z;

			int ir = 255.99f * r;
			int ig = 255.99f * g;
			int ib = 255.99f * b;
			file << ir << " " << ig << " " << ib << "\n";
		}
	}
	file.close();

	std::cout << "Finished Writing to: " << fileName << std::endl;
}