#include <GL/glew.h>
#include <glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../Constants.cuh"
#include "Texture.h"
#include <cassert>


class Window {
	GLFWwindow* window;
	int xSize, ySize;
	void initOpenGL() {
		auto t = glfwInit() == GLFW_TRUE;
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 
		glfwWindowHint(GLFW_SAMPLES, 4);
		glEnable(GL_MULTISAMPLE);
		window = glfwCreateWindow(xSize, ySize, "SVO CUDA", nullptr, nullptr);
		glfwMakeContextCurrent(window);

		glewInit();

		glViewport(0, 0, xSize, ySize);
			}

public:
	Window() : window(nullptr), xSize(0), ySize(0) { }
	Window(int xSize, int ySize) : Window() {
		this->xSize = xSize;
		this->ySize = ySize;
		initOpenGL();
	}
	~Window() {
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void draw(Texture& tex) {
		glDrawTextureNV(tex.getId(), 0, -1, -1, 1, 1, 1, 0, 0, 1, 1);
	}

	cudaGraphicsResource_t linkCUDA(Texture& tex) {
		cudaGraphicsResource_t img_GPU = 0;

		cudaError_t e;
		tex.bind();
		e = cudaGraphicsGLRegisterImage(&img_GPU, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
		return img_GPU;
	}

	cudaArray_t map(cudaGraphicsResource_t resource) {
		cudaError_t e;
		cudaArray_t prt;

		e = cudaGraphicsMapResources(1, &resource);
		assert(!e);
		e = cudaGraphicsSubResourceGetMappedArray(&prt, resource, 0, 0);
		assert_message(!e, cudaGetErrorString(e));

		return prt;
	}
	void unmap(cudaGraphicsResource_t resource) {
		cudaError_t e;
		e = cudaGraphicsUnmapResources(1, &resource);
		assert(!e);
	}

	GLFWwindow* getWindow() const { return window; }
};


