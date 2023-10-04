#pragma once
#include "Buffer.h"

class PixelBuffer : public Buffer{
	PixelBuffer() : Buffer() { }
	/// <summary>
	/// 
	/// </summary>
	/// <param name="size">size of the buffer in bytes</param>
	PixelBuffer(size_t size) : PixelBuffer() {
		glGenBuffers(1, &id);
		bind();
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STREAM_DRAW);
		unbind();
	}

	void bind() const override { glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id); }
	void unbind() const override { glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); }
private:
	GLuint id;
};
