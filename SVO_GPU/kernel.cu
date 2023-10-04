#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "3D/Ray3D.cuh"
#include "3D/RayCaster.cuh"
#include "3D/Octree3D.cuh"
#include "Camera.cuh"
#include "ImageMaker.cuh"
#include "Opengl/Window.h"
#include "OpenGL/Texture.h"
#include "Constants.cuh"

#include <stdio.h>
#include <cassert>
using namespace _3D;

__global__ void render(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch);

__global__ void render_headon(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window, double dt);

Camera mainCamera;
int main()
{
    Window window(X_RESOLUTION, Y_RESOLUTION);
    Texture texture(X_RESOLUTION, Y_RESOLUTION);
    cudaGraphicsResource_t tex_GPU = window.linkCUDA(texture);

    auto tree = Octree3D::getDefault();
    auto compiled = Octree3D::compile(&tree.front());

    mainCamera = Camera({ 0, 0, -1 });

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(X_RESOLUTION / threadsPerBlock.x, Y_RESOLUTION / threadsPerBlock.y);

    node_t* gpu_tree;
    Camera* gpu_camera;
    uchar4* gpu_result;
    size_t* gpu_pitch;
    size_t cpu_pitch;

    cudaError_t status;
    // tree
    status = cudaMalloc(&gpu_tree, sizeof(node_t) * compiled.size());
    assert(!status);
    status = cudaMemcpy(gpu_tree, compiled.data(), sizeof(node_t) * compiled.size(), cudaMemcpyHostToDevice);
    assert(!status);

    // camera
    status = cudaMalloc(&gpu_camera, sizeof(Camera));
    assert(!status);
    status = cudaMemcpy(gpu_camera, &mainCamera, sizeof(Camera), cudaMemcpyHostToDevice);
    assert(!status);

    // result
    status = cudaMallocPitch(&gpu_result, &cpu_pitch, X_RESOLUTION * sizeof(uchar4), Y_RESOLUTION);
    assert(!status);
    status = cudaMalloc(&gpu_pitch, sizeof(size_t));
    assert(!status);
    status = cudaMemcpy(gpu_pitch, &cpu_pitch, sizeof(size_t), cudaMemcpyHostToDevice);
    assert(!status);

    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);

    double lastTime = glfwGetTime();
    GLFWwindow* win = window.getWindow();
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(win, mouse_callback);
    while (!glfwWindowShouldClose(win)) {
        status = cudaMemcpy(gpu_camera, &mainCamera, sizeof(Camera), cudaMemcpyHostToDevice);
        assert(!status);

        render<<<numBlocks, threadsPerBlock>>>(gpu_camera, gpu_tree, gpu_result, gpu_pitch);
        cudaArray_t arr = window.map(tex_GPU);
        // copy        
        status = cudaMemcpy2DToArray(arr, 0, 0, gpu_result, cpu_pitch, X_RESOLUTION * sizeof(uchar4), Y_RESOLUTION, cudaMemcpyDefault);
        // status = cudaMemcpyToArray(arr, 0, 0, gpu_result, s, cudaMemcpyDefault);
        window.unmap(tex_GPU);
        status = cudaDeviceSynchronize();
        assert(!status);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture.getId(), 0);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // if not already bound
        glBlitFramebuffer(0, 0, X_RESOLUTION, Y_RESOLUTION, 0, 0, X_RESOLUTION, Y_RESOLUTION, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        // window.draw(texture);


        glfwSwapBuffers(win);
        glfwPollEvents();

        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;
        int fps = 1. / delta;
        lastTime = currentTime;
        std::cout << "FPS: " << fps << '\n';
        auto& p = mainCamera.Position;

        std::cout << p.x << ", " << p.y << ", " << p.z << "\n";

        processInput(win, delta);
    }

    glDeleteFramebuffers(1, &fbo);

    cudaFree(gpu_tree);
    cudaFree(gpu_camera);
    cudaFree(gpu_result);
    cudaFree(gpu_pitch);

    // SaveImage("out.ppm", X_RESOLUTION, Y_RESOLUTION, data);
    return 0;
}

__global__ void render(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch) {
    const uchar4 CLEAR_COLOUR = make_uchar4(127, 127, 127, 127);
    const float3 lower_left = make_float3(-1, -1, 1);
    const float3 span = make_float3(2, 2, 0);
    const float3 camPos = make_float3(camera->Position);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = float(x) / float(X_RESOLUTION);
    float v = float(y) / float(Y_RESOLUTION);
    float3 rayDir = make_float3(u, v, 0) * span + lower_left;

    glm::vec3 d_ = glm::normalize(make_vec3(rayDir));
    d_ = camera->getRotation() * d_;
    // rayDir = make_float3(d_);

    _3D::Ray3D ray(camPos, rayDir);

    auto res = castRay(ray, tree);

    if (y > 550) {
        res = castRay(ray, tree);
    }

    uchar4* d = element(data, *pitch, x, y);
    unsigned char& r = d->x;
    unsigned char& g = d->y;
    unsigned char& b = d->z;
    unsigned char& a = d->w;
    if (res.hit) {
        r = 255;
        g = 0;
        b = 0;
        a = 255;
    }
    else {
        *d = CLEAR_COLOUR;
    }
    if (x == X_RESOLUTION / 2) {
        r = 0;
        g = 255;
        b = 0;
    }
    if (y == Y_RESOLUTION / 2) {
        r = 0;
        g = 255;
        b = 0;
    }
}

__global__ void render_headon(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch) {
    const uchar4 CLEAR_COLOUR = make_uchar4(127, 127, 127, 127);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = float(x) / float(X_RESOLUTION);
    float v = float(y) / float(Y_RESOLUTION);

    float3 rayPos = make_float3(u * 8, v * 8, -1);
    float3 rayDir = make_float3(0, 0, 1);
    rayPos.x += camera->Position.x;
    rayPos.y += camera->Position.y;
    rayPos.z += camera->Position.z;

    if (x > 80) {
        int i = 0;
        auto d = element(data, *pitch, x, y);
    }

    _3D::Ray3D ray(rayPos, rayDir);

    auto res = castRay(ray, tree);

    uchar4* d = element(data, *pitch, x, y);
    unsigned char& r = d->x;
    unsigned char& g = d->y;
    unsigned char& b = d->z;
    unsigned char& a = d->w;
    if (res.hit) {
        r = 255;
        g = 0;
        b = 0;
        a = 255;
    }
    else {
        *d = CLEAR_COLOUR;
    }
}

void processInput(GLFWwindow* window, double dt)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        mainCamera.ProcessKeyboard(FORWARD, dt);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        mainCamera.ProcessKeyboard(BACKWARD, dt);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        mainCamera.ProcessKeyboard(LEFT, dt);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        mainCamera.ProcessKeyboard(RIGHT, dt);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        mainCamera.ProcessKeyboard(UP, dt);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        mainCamera.ProcessKeyboard(DOWN, dt);
}

bool firstMouse = true;
float lastX = X_RESOLUTION * .5f;
float lastY = X_RESOLUTION * .5f; 
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    mainCamera.ProcessMouseMovement(xoffset, yoffset);
}
