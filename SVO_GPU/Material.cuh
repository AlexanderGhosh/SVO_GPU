#pragma once
#include "Constants.cuh"


struct Material {
	uchar3 diffuse;
	float diffuseC;
	float specularC;

	Material() : diffuse(), diffuseC(0), specularC(0) { }
	Material(uchar3 colour, float diffuse, float specular) : diffuse(colour), diffuseC(diffuse), specularC(specular) { }
};