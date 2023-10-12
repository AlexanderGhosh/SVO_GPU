#include "Constants.cuh"
#include "VectorMaths.cuh"
#include "3D/Ray3D.cuh"
#include "math_functions.h"
#include "Material.cuh"

struct CastResult;
struct Model_;

__host__ __device__ CastResult castRay(const _3D::Ray3D& ray, const Model_ model, float& tprev);

struct StackItem_ {
	int32_t parentIdx;
	uint8_t slot;
};
class Stack_ {
public:
	__host__ __device__ Stack_() : data_(), size_(0) { }
	__host__ __device__ void push(int32_t parent, uint8_t slot) { data_[size_++] = { parent, slot }; }
	__host__ __device__ StackItem_& pop() { return data_[--size_]; }
	__host__ __device__ const uint8_t size() const { return size_; };
private:
	StackItem_ data_[3];
	uint8_t size_;
};

struct CastResult {
	bool hit;
	float3 hitPos;
	float3 faceNormal;
	uint8_t materialIndex;
};

struct ModelDetails {
	float3 position;
	float4 rotation;

	float2 span;

	__host__ __device__ ModelDetails() : position(make_float3(0, 0, 0)), rotation(make_float4(0, 0, 0, 0)), span(make_float2(0, 0)) { }
};

struct Model_ {
	ModelDetails details;
	node_t* tree;
	Material* materials;
};

__host__ __device__ float3 toColour(uchar3 c) {
	return make_float3(c.x / 255.9f, c.y / 255.9f, c.z / 255.9f);
}

__host__ __device__ uchar3 fromColour(float3 c) {
	return make_uchar3(c.x * 255.9f, c.y * 255.9f, c.z * 255.9f);
}

__host__ __device__ uchar4 bling_phong(const Material& mat, const CastResult& dets, const float3& lightDir, const _3D::Ray3D& ray) {
	const float3 c = toColour(mat.diffuse);
	float3 amb = c * AMBIENT;

	float ln = clamp(0, 1, dot(lightDir, dets.faceNormal));
	float3 dif = c * ln * mat.diffuseC;

	float3 r = reflect(lightDir, dets.faceNormal);
	float3 v = -ray.getDirection();

	float rv = clamp(0, 1, dot(r, v));

	float spc_ = powf(rv, SPECULAR_ALPHA) * mat.specularC;
	float3 spc = make_float3(spc_, spc_, spc_);

	float3 s = amb + dif + spc;
	uchar3 t_ = fromColour(s);
	uchar4 res{};
	res.x = t_.x;
	res.y = t_.y;
	res.z = t_.z;
	res.w = 255;
	return res;
}


__global__ void render_new(const Camera* camera, node_t* all_trees, uint32_t* treeSizes, float3* treePositions, uint32_t numTrees, Material* materials, ModelDetails* details, size_t* pitch, uchar4* out) {

	const uchar4 CLEAR_COLOUR = make_uchar4(127, 127, 127, 255);
	const float focal_length = 1;
	const float3 lower_left = make_float3(-1, -1, 0);
	const float3 span = make_float3(2, 2, 0);
	const float3 camPos = make_float3(camera->Position);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


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
	float3 f = make_float3(camera->Front);
	float3 image_point = ri + u + camPos + f * focal_length;

	float3 ray_direction = image_point - camPos;


	_3D::Ray3D ray(camPos, ray_direction);

	float tprev = FLT_MAX;
	CastResult res{};
	uint32_t idx = 0;
	Model_ m;
	m.materials = materials;
	m.details = {};
	m.details.span = { .25f, 2.f };
	for (uint32_t i = 0; i < numTrees; i++) {
		m.tree = &all_trees[idx];
		m.details.position = treePositions[i];
		float t = 0;
		auto r = castRay(ray, m, t);
		if (t < tprev && r.hit) {
			res = r;
			tprev = t;
		}
		idx += treeSizes[i];
	}
	uchar4* d = element(out, *pitch, x, y);
	unsigned char& r = d->x;
	unsigned char& g = d->y;
	unsigned char& b = d->z;
	unsigned char& a = d->w;
	if (res.hit) {
		const Material& mat = materials[res.materialIndex];
		*d = bling_phong(mat, res, -normalize(make_float3(1, -1, 1)), ray);
		a = 255;
	}
	else {
		*d = CLEAR_COLOUR;
	}
}

void test(tree_t tree) {
	const float3 camPos = make_float3(0, 0, -1);
	const float3 right = make_float3(-1, 0, 0);
	const float3 up = make_float3(0, 1, 0);
	const float3 front = make_float3(0, 0, 1);

	const float width = X_RESOLUTION;  // pixels across
	const float height = Y_RESOLUTION;  // pixels high

	Model_ m;
	m.tree = tree.data();
	m.details = ModelDetails();
	m.details.span = { 1, 8 };
	float tmax = 0;

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			float normalized_i = (x / width) - 0.5;
			float normalized_j = (y / height) - 0.5;
			float3 ri = right;
			ri.x *= -normalized_i;
			ri.y *= -normalized_i;
			ri.z *= -normalized_i;
			float3 u = up;
			u.x *= normalized_j;
			u.y *= normalized_j;
			u.z *= normalized_j;
			float3 image_point = ri + u + camPos + front * 1;

			float3 ray_direction = image_point - camPos;


			_3D::Ray3D ray(camPos, ray_direction);
			

			auto res = castRay(ray, m, tmax);

			if (res.hit) {
			}
		}
	}

}

__host__ __device__ __inline__ float3 getChildFront(float3 pf, uint8_t slot, float scale) {
	pf.x += slot & 1 ? scale : 0;
	pf.y += slot & 2 ? scale : 0;
	pf.z += slot & 4 ? scale : 0;
	return pf;
}

__host__ __device__ int getChildIndex_(const uint32_t parent, const uint8_t child_slot)
{
	const int child_idx = (parent & 0xffff0000) >> 16;
	const uint32_t valid_mask = 1 << child_slot + 8;
	if (parent & valid_mask) {
		const uint32_t valid = (parent & 0x0000ff00) >> 8;
		const uint32_t leafs = (parent & 0x000000ff);
		const uint32_t not_leafs = valid ^ leafs;
		int offset = 0;
		for (int i = 0; i < 8; i++) {
			if ((not_leafs >> i) & 1) {
				if (i == child_slot) {
					break;
				}
				offset++;
			}
		}
		return child_idx + offset;
	}
	return -1;
}

__host__ __device__ float3 getNormal(const float tmin, const float3& t_cf, const uint8_t mirrorMask) {
	float3 norm = make_float3(0, 0, 0);
	norm.x = t_cf.x == tmin ? (mirrorMask & 1 ? -1 : 1) : 0;
	norm.y = t_cf.y == tmin ? (mirrorMask & 2 ? -1 : 1) : 0;
	norm.z = t_cf.z == tmin ? (mirrorMask & 4 ? -1 : 1) : 0;
	// if (t_cf.x == tmin) norm.x = 1;
	// if (t_cf.y == tmin) norm.y = 1;
	// if (t_cf.z == tmin) norm.z = 1;
	// if (mirrorMask & 1) norm.x *= -1;
	// if (mirrorMask & 2) norm.y *= -1;
	// if (mirrorMask & 4) norm.z *= -1;
	return normalize(norm);
}

__host__ __device__ uint8_t getMaterialIndex(const node_t& parent, const uint8_t child_slot)
{
	uint32_t shift = child_slot * 4;
	return (parent.shader_data >> shift) & 15;
}

__host__ __device__ CastResult castRay(const _3D::Ray3D& ray, const Model_ model, float& tprev) {
	CastResult result{};
	result.hit = false;

	const node_t* tree = model.tree;
	const uint32_t minScale = model.details.span.x;
	const uint32_t maxScale = model.details.span.y;

	float scale = maxScale;
	float half = scale * .5f;

	node_t parent = tree[0];
	Stack_ parent_stack; // allows for a maximum scale of 64

	float3 pmin = make_float3(0, 0, 0);
	// float3 mirroredOrigin = make_float3(0, 0, 0);
	float3 rayPos = ray.getPos() - model.details.position;

	uint8_t mirrorMask = 0;
	if (ray.dir_inv_.x < 0) {
		mirrorMask ^= 1; 
		rayPos.x = 2 * scale - rayPos.x;
		pmin.x += scale;
	}
	if (ray.dir_inv_.y < 0) {
		mirrorMask ^= 2;
		rayPos.y = 2 * scale - rayPos.y;
		pmin.y += scale;
	}
	if (ray.dir_inv_.z < 0) {
		mirrorMask ^= 4;
		rayPos.z = 2 * scale - rayPos.z;
		pmin.z += scale;
	}
	float3 mirroredOrigin = pmin;

	float3 inv_ray_dir = abs(ray.dir_inv_);

	// if (mirrorMask & 1) rayPos.x = 2 * pmin.x - rayPos.x;
	// if (mirrorMask & 2) rayPos.y = 2 * pmin.y - rayPos.y;
	// if (mirrorMask & 4) rayPos.z = 2 * pmin.z - rayPos.z;

	auto tFunc = [&](const float3& a) -> const float3& {
		return (a - rayPos) * inv_ray_dir;
	};

	// mirrorMask = 0;

	float tmin = 0;
	float tmax = 0;

	// float3 pmin = mirroredOrigin;
	float3 pcen = pmin + half;
	float3 pmax = pmin + scale;

	float3 t_pf = tFunc(pmin);
	float3 t_pc = tFunc(pcen);
	float3 t_pb = tFunc(pmax);

	tmin = max(t_pf);
	tmax = min(t_pb);

	if (tmin >= tmax) {
		return result;
	}
	if (signbit(t_pf.x) && signbit(t_pb.x)) {
		return result;
	}
	if (signbit(t_pf.y) && signbit(t_pb.y)) {
		return result;
	}
	if (signbit(t_pf.z) && signbit(t_pb.z)) {
		return result;
	}


	uint8_t childSlot = 0;
	// if (t_pc.x <= tmin) childSlot ^= 1;
	// if (t_pc.y <= tmin) childSlot ^= 2;
	// if (t_pc.z <= tmin) childSlot ^= 4;

	childSlot ^= t_pc.x <= tmin ? 1 : 0;
	childSlot ^= t_pc.y <= tmin ? 2 : 0;
	childSlot ^= t_pc.z <= tmin ? 4 : 0;

	childSlot %= 7;

	scale = half;
	half *= 0.5f;

	bool _;
	int32_t idx = 0;
	uint8_t ittertion = 0;
	bool forcePop = false;
	uint32_t delta = 0;
	float3 cmin;
	float3 t_cf;
	float3 ccen;
	float3 t_cc;
	float3 cmax;
	float3 t_cb;
	float tc_max;
	uint8_t mirroredChildSlot;
	uint8_t validMask;
	//float3 t_pb;
	uint8_t prevSlot;
	StackItem_ item;
	while (ittertion++ < MAX_ITTERATIONS) {
		parent = tree[idx];

		cmin = getChildFront(pmin, childSlot, scale);
		t_cf = tFunc(cmin);
		tmin = max(t_cf);

		mirroredChildSlot = childSlot ^ mirrorMask;
		validMask = parent.child_data >> 8;
		if (validMask >> mirroredChildSlot & 0x1 && !forcePop) {
			// child is valid
			if (parent.child_data >> mirroredChildSlot & 0x1 || half < minScale) {
				// is leaf

				result.hit = true;
				result.hitPos = ray.point(tmin) + model.details.position; // needs to be mirrored (i dont think so actually)
				result.faceNormal = getNormal(tmin, t_cf, mirrorMask);
				result.materialIndex = getMaterialIndex(parent, childSlot ^ mirrorMask);
				tprev = tmin;
				return result;
			}


			// PUSH
			/*if (tc_max < tmax) {
			* can prevent the pushing parents which will just be popped anyways
			}*/
			parent_stack.push(idx, childSlot);
			idx = getChildIndex_(parent.child_data, mirroredChildSlot);
			parent = tree[idx];


			ccen = cmin + half;
			t_cc = tFunc(ccen);

			// the next child
			childSlot = 0;
			// if (t_cc.x <= tmin) childSlot ^= 1;
			// if (t_cc.y <= tmin) childSlot ^= 2;
			// if (t_cc.z <= tmin) childSlot ^= 4;

			childSlot ^= t_cc.x <= tmin ? 1 : 0;
			childSlot ^= t_cc.y <= tmin ? 2 : 0;
			childSlot ^= t_cc.z <= tmin ? 4 : 0;
			// childSlot %= 7;

			pmin = cmin;
			scale = half;
			half *= .5f;
			t_pf = tFunc(pmin);
			tmin = max(t_pf);

			// pmax = pmin + scale;
			// t_pb = tFunc(pmax);
			// tmax = min(t_pb);

			continue;
			
			// get centre t
			// if less than child back
			// PUSH
			// else
			// POP or ADVANCE
		}
		// tmax = tc_max;

		cmax = cmin + scale;
		t_cb = tFunc(cmax);
		tc_max = min(t_cb);

		prevSlot = childSlot;
		// if (t_cb.x <= tc_max) childSlot ^= 1;
		// if (t_cb.y <= tc_max) childSlot ^= 2;
		// if (t_cb.z <= tc_max) childSlot ^= 4;
		childSlot ^= t_cb.x <= tc_max ? 1 : 0;
		childSlot ^= t_cb.y <= tc_max ? 2 : 0;
		childSlot ^= t_cb.z <= tc_max ? 4 : 0;

		if (forcePop || prevSlot & ~childSlot) {
			// POP
			if (!parent_stack.size()) {
				return result;
			}

			delta = forcePop ? delta : prevSlot ^ childSlot;
			item = parent_stack.pop();
			idx = item.parentIdx;
			childSlot = item.slot ^ delta;

			forcePop = delta & item.slot;
			// if (delta & item.slot) {
			// 	 // will be called incorreclty if you need to pop more than once in a row
			// 	 forcePop = true;
			// 	 // return result;
			// }

			// if (item.slot & 1) pmin.x -= scale * 2.f;
			// if (item.slot & 2) pmin.y -= scale * 2.f;
			// if (item.slot & 4) pmin.z -= scale * 2.f;

			half = scale;
			scale *= 2;

			pmin.x -= item.slot & 1 ? scale : 0;
			pmin.y -= item.slot & 2 ? scale : 0;
			pmin.z -= item.slot & 4 ? scale : 0;


			// t_pf = tFunc(pmin);
			// tmin = max(t_pf);
		}
	}

	// if tmin < tmax then it hits
	return result;
}

// change stack to be indexed by scale, and poping will get the closest scale that is >= requested