#pragma once
#include "GL/glew.h"

class Buffer {
public:
	Buffer() : id(0) { }
	virtual ~Buffer() { glDeleteBuffers(1, &id); }
	virtual void bind() const = 0;
	virtual void unbind() const = 0;
protected:
	GLuint id;
};
