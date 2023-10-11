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

#include "ModelLoader/QB_ModelLoader.h"
#include "ModelLoader/Model.h"
#include "NewRayCaster.cuh"

#include <stdio.h>
#include <cassert>
using namespace _3D;

__global__ void render(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch, material_t* materials);
__global__ void render_headon(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch);

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window, double dt);

Camera mainCamera;
int main()
{
    QB_Loader modelLoader;
    Model model = modelLoader.load("C:\\Users\\AGWDW\\Desktop\\test2.qb");
    ModelDetails dets;
    dets.span = { 1, 8 };

    test(tree_t(model.getData(), model.getData()+model.getSize()));

    Window window(X_RESOLUTION, Y_RESOLUTION);

    Texture texture(X_RESOLUTION, Y_RESOLUTION);
    cudaGraphicsResource_t tex_GPU = window.linkCUDA(texture);

    const uint8_t num_shaders = 5;
    material_t shaders[num_shaders] = {
        { 0, 0, 0, 255 },
        { 255, 255, 255, 255 },
        { 255, 0, 0, 255 },
        { 0, 255, 0, 255 },
        { 0, 0, 255, 255 },
    };

    // auto tree = Octree3D::getDefault();
    // auto compiled = Octree3D::compile(&tree.front());

    // auto compiled = model.getData();

    mainCamera = Camera({ 0, 0, -1 });

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(X_RESOLUTION / threadsPerBlock.x, Y_RESOLUTION / threadsPerBlock.y);

    node_t* gpu_tree;
    Camera* gpu_camera;
    uchar4* gpu_result;
    size_t* gpu_pitch;
    size_t cpu_pitch;
    material_t* gpu_materials;
    ModelDetails* gpu_modelDetails;

    cudaError_t status;
    // tree
    status = cudaMalloc(&gpu_tree, sizeof(node_t) * model.getSize());
    assert(!status);
    status = cudaMemcpy(gpu_tree, model.getData(), sizeof(node_t) * model.getSize(), cudaMemcpyHostToDevice);
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

    // shaders
    status = cudaMalloc(&gpu_materials, sizeof(material_t) * num_shaders);
    assert(!status);
    status = cudaMemcpy(gpu_materials, &shaders, sizeof(material_t) * num_shaders, cudaMemcpyHostToDevice);
    assert(!status);

    // modelDetails
    status = cudaMalloc(&gpu_modelDetails, sizeof(ModelDetails));
    assert(!status);
    status = cudaMemcpy(gpu_modelDetails, &dets, sizeof(ModelDetails), cudaMemcpyHostToDevice);
    assert(!status);

    // model
    // status = cudaMalloc(&gpu_model, sizeof(Model));
    // assert(!status);
    // status = cudaMemcpy(gpu_model, &model, sizeof(Model), cudaMemcpyHostToDevice);
    // assert(!status);


    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);

    double lastTime = glfwGetTime();
    GLFWwindow* win = window.getWindow();
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(win, mouse_callback);
    cudaEvent_t start, stop;
    float totalTime = 0;
    uint32_t frameCount = 0;
    while (!glfwWindowShouldClose(win)) {
        status = cudaMemcpy(gpu_camera, &mainCamera, sizeof(Camera), cudaMemcpyHostToDevice);
        assert(!status);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        render_new<<<numBlocks, threadsPerBlock>>>(gpu_camera, gpu_tree, gpu_materials, gpu_modelDetails, gpu_pitch, gpu_result);
        // render<<<numBlocks, threadsPerBlock>>>(gpu_camera, gpu_tree, gpu_result, gpu_pitch, gpu_materials);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop); 
        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
        frameCount++;
        

        //render_headon << <numBlocks, threadsPerBlock >> > (gpu_camera, gpu_tree, gpu_result, gpu_pitch);
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
        // std::cout << "FPS: " << fps << '\n';

        processInput(win, delta);
    }

    printf("Time to generate:  %3.1f ms \n", totalTime / (float)frameCount);

    glDeleteFramebuffers(1, &fbo);

    cudaFree(gpu_tree);
    cudaFree(gpu_camera);
    cudaFree(gpu_result);
    cudaFree(gpu_pitch);

    // SaveImage("out.ppm", X_RESOLUTION, Y_RESOLUTION, data);
    return 0;
}

__global__ void render(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch, material_t* materials) {
    const uchar4 CLEAR_COLOUR = make_uchar4(127, 127, 127, 255);
    const float focal_length = 1;
    const float3 camPos = make_float3(camera->Position);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    x = 500;
    y = 600;

    const float width = X_RESOLUTION;  // pixels across
    const float height = Y_RESOLUTION;  // pixels high
    float normalized_i = (x / width) - 0.5;
    float normalized_j = (y / height) - 0.5;
    float3 ri = make_float3(camera->Right);
    ri.x *= -normalized_i;
    ri.y *= -normalized_i;
    ri.z *= -normalized_i;
    float3 u = make_float3(camera->Up);
    u.x *= normalized_j;
    u.y *= normalized_j;
    u.z *= normalized_j;
    float3 image_point = ri + u + camPos + make_float3(camera->Front) * focal_length;

    float3 ray_direction = image_point - camPos;


    _3D::Ray3D ray(camPos, ray_direction);

    auto res = castRay(ray, tree);
    ray_direction = ray.getDirection();

    uchar4* d = element(data, *pitch, x, y);
    unsigned char& r = d->x;
    unsigned char& g = d->y;
    unsigned char& b = d->z;
    unsigned char& a = d->w;
    if (res.hit) {
        const float3 light_dir = normalize(make_float3(1, -1, 1));
        const material_t mat = materials[res.material_index];
        float angle = light_dir.x * res.normal.x + light_dir.y * res.normal.y + light_dir.z * res.normal.z;
        angle = clamp(EPSILON, 1, angle);
        angle = 1;
        r = ((float) mat.x) * angle;
        g = ((float) mat.y) * angle;
        b = ((float) mat.z) * angle;

       // r = (res.normal.x * 0.5f + 0.5f) * 255.9;
       // g = (res.normal.y * 0.5f + 0.5f) * 255.9;
       // b = (res.normal.z * 0.5f + 0.5f) * 255.9;
       // r = 255;
       // g = 255;
       // b = 255;
        a = 255;
    }    
    else {
        *d = CLEAR_COLOUR;
    }
    if (x > 500 && x < 510 && y > 600 && y < 610) {
        g = 255;
    }
}

__global__ void render_headon(const Camera* camera, node_t* tree, uchar4* data, size_t* pitch) {
    const uchar4 CLEAR_COLOUR = make_uchar4(127, 127, 127, 127);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = float(x) / float(X_RESOLUTION);
    float v = float(y) / float(Y_RESOLUTION);

    /*float3 rayPos = make_float3(u * 8, v * 8, -1);
    float3 rayDir = make_float3(0, 0, 1);
    rayPos.x += camera->Position.x;
    rayPos.y += camera->Position.y;
    rayPos.z += camera->Position.z;

    _3D::Ray3D ray(rayPos, rayDir);*/

    float3 rayPos = make_float3(u * 8, v * 8, 1);
    float3 rayDir = make_float3(0, 0, 1);
    // rayPos.x += camera->Position.x;
    // rayPos.y += camera->Position.y;
    // rayPos.z += camera->Position.z;

    _3D::Ray3D ray(make_float3(0, 0, -1), rayPos);

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
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    mainCamera.ProcessMouseMovement(xoffset, yoffset);
    firstMouse = false;
}
