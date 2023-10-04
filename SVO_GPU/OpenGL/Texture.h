#pragma once
#include "GL/glew.h"

class Texture {
public:
	Texture() : texture(0) { }
	Texture(int width, int height) : Texture() {
        glGenTextures(1, &texture);
        bind();

        // set basic parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Create texture data (4-component unsigned byte)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // Unbind the texture
        unbind();
	}

    ~Texture() {
        glDeleteTextures(1, &texture);
    }

    void bind() { glBindTexture(GL_TEXTURE_2D, texture); }
    void unbind() { glBindTexture(GL_TEXTURE_2D, 0); }

    const GLuint getId() const { return texture; }
private:
	GLuint texture;
};