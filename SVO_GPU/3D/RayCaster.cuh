#pragma once
#include "Ray3D.cuh"
#include "cuda_runtime.h"
#include "../VectorMaths.cuh"
#include "../Constants.cuh"

namespace _3D {
	struct RayCastResult3D {
		Ray3D ray;
		bool hit;
		float3 pos;
		float scale;
		uint8_t shader_index;

		__device__ RayCastResult3D();
	};

	struct StackItem {
		node_t parent;
		slot_t slot;
	};
	class Stack {
	public:
		__device__ Stack() : data(), size(0) { } 
		__device__ void push(node_t parent, slot_t slot) { data[size++] = { parent, slot }; }
		__device__ StackItem& pop() { return data[--size]; }
	private:
		StackItem data[PARENT_STACK_DEPTH + 1];
		int size;
	};


	/// <summary>
	/// Will return the next child slot to investigate (PUSH into)
	/// </summary>
	/// <param name="t_front">the t-value of the next parent's front</param>
	/// <param name="t_centre">the t-value of the next parent's centre</param>
	/// <returns>the next child slot to PUSH into</returns>
	__device__ __inline__ slot_t push(const float3& t_front, const float3& t_centre);
	/// <summary>
	/// Returns the next child slot to investigate of the same parent or will set should_pop=true
	/// </summary>
	/// <param name="slot">the current child slot</param>
	/// <param name="t_back">the t-value of the child</param>
	/// <param name="should_pop">true if there is no sibling to ADVANCE to | otherwise false</param>
	/// <returns>the slot of the sibling to ADVANCE to</returns>
	__device__ __inline__ slot_t advance(slot_t slot, const float3& t_back, bool& should_pop);
	/// <summary>
	/// Returns the front position of the next child in the given slot
	/// </summary>
	/// <param name="parent_front">the front pos of the desired parent</param>
	/// <param name="next_slot">the slot the child will take on</param>
	/// <param name="next_scale">the scale that the child will posses</param>
	/// <returns>the front position for the appropriate child</returns>
	__device__ __inline__ float3 getChildPosition(float3 parent_front, const slot_t next_slot, const float next_scale);
	/// <summary>
	/// Will mirror th eslot if appropriate
	/// </summary>
	/// <param name="slot">the original un altered slot</param>
	/// <param name="mirror">the mirror</param>
	/// <returns>a valid slot mirrored correctly</returns>
	__device__ __inline__ slot_t mirrorSlot(slot_t slot, const mirror_t& mirror);
	/// <summary>
	/// Returns the front of the octree when mirrored
	/// </summary>
	/// <param name="orign">the original location of the octree</param>
	/// <param name="scale">the scale of the octee</param>
	/// <param name="mirror">the mirror</param>
	/// <returns>The front of the mirroed octree (the mirror line)</returns>
	__device__ __inline__ float3 getMirrorLine(const float3& orign, float scale, const mirror_t& mirror);
	/// <summary>
	/// Mirrors pos around the mirror line
	/// </summary>
	/// <param name="pos">the position to be mirrored</param>
	/// <param name="mirror_line">the line to mirror around</param>
	/// <param name="mirror">the mirror</param>
	/// <returns>the position mirrored around mirror_line</returns>
	__device__ __inline__ float3 mirrorAround(float3 pos, const float3& mirror_line, const mirror_t& mirror);
	/// <summary>
	/// Returns the index of the child -1 if it isnt a valid is_leaf is true if it is valid and a leaf
	/// </summary>
	/// <param name="parent">the current parent</param>
	/// <param name="child_slot">the desired slot for the index</param>
	/// <param name="is_leaf">true if the slot is also a child | othewise false</param>
	/// <returns>the index of the child</returns>
	__device__ __inline__ int getChildIndex(const uint32_t parent, const slot_t child_slot, bool& is_leaf);
	/// <summary>
	/// Returns the index of the child -1 if it isnt a valid is_leaf is true if it is valid and a leaf
	/// </summary>
	/// <param name="parent">the current parent</param>
	/// <param name="child_slot">the desired slot for the index</param>
	/// <param name="mirror">the mirror</param>
	/// <param name="is_leaf">true if the slot is also a child | othewise false</param>
	/// <returns>the index of the child</returns>
	__device__ __inline__ int getChildIndex(const uint32_t parent, slot_t child_slot, const mirror_t& mirror, bool& is_leaf);
	/// <summary>
	/// Assures the ray's direction is strictly positive | and changes the position appropriatly
	/// </summary>
	/// <param name="ray">the unaltered ray</param>
	/// <param name="mirror_line">the line to mirror arround</param>
	/// <param name="mirror">the mirror</param>
	/// <returns>the ray subtible for a mirrord octree</returns>
	__device__ __inline__ Ray3D correctRay(const Ray3D& ray, const float3& mirror_line, const mirror_t& mirror);

	__device__ bool hits_at_all(const Ray3D& ray, const float3 front, const float3 back);

	__device__ uint8_t getShaderIndex(const node_t& parent, const slot_t child_slot, const mirror_t& mirror);

	__device__ RayCastResult3D castRay(const Ray3D& ray, node_t* tree);
}