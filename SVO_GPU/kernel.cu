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


void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window, double dt);

Camera mainCamera;
int main()
{
    QB_Loader modelLoader;
    Model model = modelLoader.load("C:\\Users\\AGWDW\\Desktop\\test2.qb");
    ModelDetails dets;
    dets.span = { .25f, 2.f };
    dets.position = make_float3(0, 0, 0);

    // test(tree_t(model.getData(), model.getData() + model.getTotalSize()));

    Window window(X_RESOLUTION, Y_RESOLUTION);

    Texture texture(X_RESOLUTION, Y_RESOLUTION);
    cudaGraphicsResource_t tex_GPU = window.linkCUDA(texture);

    Material shaders[MATERIAL_COUNT] = {
        Material(make_uchar3(0, 0, 0),     0.5f, 0.07f),      
        Material(make_uchar3(255, 255, 0), 0.5f, 0.07f),
        Material(make_uchar3(255, 0, 0),   0.5f, 0.07f),
        Material(make_uchar3(0, 255, 0),   0.5f, 0.07f),
        Material(make_uchar3(0, 0, 255),   0.5f, 0.07f)
    };

    // auto tree = Octree3D::getDefault();
    // auto compiled = Octree3D::compile(&tree.front());

    // auto compiled = model.getData();

    mainCamera = Camera({ 0, 0, -1 });

    dim3 threadsPerBlock(25, 25);
    dim3 numBlocks(X_RESOLUTION / threadsPerBlock.x, Y_RESOLUTION / threadsPerBlock.y);

    node_t* gpu_trees;
    uint32_t* gpu_treeSizes;
    float3* gpu_treePositions;
    Camera* gpu_camera;
    uchar4* gpu_result;
    size_t* gpu_pitch;
    size_t cpu_pitch;
    Material* gpu_materials;
    ModelDetails* gpu_modelDetails;

    cudaError_t status;
    // trees
    status = cudaMalloc(&gpu_trees, sizeof(node_t) * model.getTotalSize());
    assert(!status);
    status = cudaMemcpy(gpu_trees, model.getData(), sizeof(node_t) * model.getTotalSize(), cudaMemcpyHostToDevice);
    assert(!status);
    // tree sizes
    const auto sizes = model.getTreeSizes();
    status = cudaMalloc(&gpu_treeSizes, sizeof(uint32_t) * sizes.size());
    assert(!status);
    status = cudaMemcpy(gpu_treeSizes, sizes.data(), sizeof(uint32_t) * sizes.size(), cudaMemcpyHostToDevice);
    assert(!status);
    // tree positions
    const auto positions = model.getTreePositions();
    status = cudaMalloc(&gpu_treePositions, sizeof(float3) * positions.size());
    assert(!status);
    status = cudaMemcpy(gpu_treePositions, positions.data(), sizeof(float3) * positions.size(), cudaMemcpyHostToDevice);
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
    status = cudaMalloc(&gpu_materials, sizeof(Material) * MATERIAL_COUNT);
    assert(!status);
    status = cudaMemcpy(gpu_materials, &shaders, sizeof(Material) * MATERIAL_COUNT, cudaMemcpyHostToDevice);
    assert(!status);

    // modelDetails
    status = cudaMalloc(&gpu_modelDetails, sizeof(ModelDetails));
    assert(!status);
    status = cudaMemcpy(gpu_modelDetails, &dets, sizeof(ModelDetails), cudaMemcpyHostToDevice);
    assert(!status);

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
        // const Camera* camera, node_t* all_trees, uint32_t* treeSizes, uint32_t numTrees, Material* materials, ModelDetails* details, size_t* pitch, uchar4* out
        render_new<<<numBlocks, threadsPerBlock>>>(gpu_camera, gpu_trees, gpu_treeSizes, gpu_treePositions, sizes.size(), gpu_materials, gpu_modelDetails, gpu_pitch, gpu_result);
        // render<<<numBlocks, threadsPerBlock>>>(gpu_camera, gpu_trees, gpu_result, gpu_pitch, gpu_materials);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop); 
        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
        frameCount++;
        

        //render_headon << <numBlocks, threadsPerBlock >> > (gpu_camera, gpu_trees, gpu_result, gpu_pitch);
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

    cudaFree(gpu_trees);
    cudaFree(gpu_treeSizes);
    cudaFree(gpu_camera);
    cudaFree(gpu_result);
    cudaFree(gpu_pitch);

    // SaveImage("out.ppm", X_RESOLUTION, Y_RESOLUTION, data);
    return 0;
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
