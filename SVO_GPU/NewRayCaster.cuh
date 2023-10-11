#include "Constants.cuh"
#include "VectorMaths.cuh"
#include "3D/Ray3D.cuh"

struct CastResult;
struct Model_;

__host__ __device__ CastResult castRay(const _3D::Ray3D& ray, const Model_ model, float& tprev);

struct StackItem_ {
	int32_t parentIdx;
	uint8_t slot;
	__host__ __device__ StackItem_() : parentIdx(-1), slot(0) { }
	__host__ __device__ StackItem_(int32_t p, uint8_t s) : parentIdx(p), slot(s) { }
};
template<uint8_t length>
class Stack_ {
public:
	__host__ __device__ Stack_() : data(), size(0) { }
	__host__ __device__ void push(int32_t parent, uint8_t slot, uint8_t scale) { item(scale) = {parent, slot}; }
	__host__ __device__ StackItem_ pop(uint8_t& scale) 
	{ 
		uint8_t idx = index(scale);
		auto res = pop_index(idx);
		scale = powf(2, idx);
		return res;
	}
private:
	__host__ __device__ StackItem_ pop_index(uint8_t& index)
	{
		auto res = data[index];
		res = res.parentIdx >= 0 ? res : pop_index(++index);
		data[index].parentIdx = -1;
		return res;
	}
	__host__ __device__ uint8_t index(float scale) {
		return log2f(scale);
	}
	__host__ __device__ StackItem_& item(float scale) {
		return data[index(scale)];
	}
	StackItem_ data[length];
	int size;
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

	uint2 span;
};

struct Model_ {
	ModelDetails details;
	node_t* tree;
	material_t* materials;
};

__global__ void render_new(const Camera* camera, node_t* tree, material_t* materials, ModelDetails* details, size_t* pitch, uchar4* out) {
	Model_ m;
	m.tree = tree;
	m.materials = materials;
	if (details) m.details = *details;

	const uchar4 CLEAR_COLOUR = make_uchar4(127, 127, 127, 255);
	const float focal_length = 1;
	const float3 lower_left = make_float3(-1, -1, 0);
	const float3 span = make_float3(2, 2, 0);
	const float3 camPos = make_float3(camera->Position);

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// x = 500;
	// y = 600;

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

	float tprev = 0;
	auto res = castRay(ray, m, tprev);

	uchar4* d = element(out, *pitch, x, y);
	unsigned char& r = d->x;
	unsigned char& g = d->y;
	unsigned char& b = d->z;
	unsigned char& a = d->w;
	if (res.hit) {
		const float3 light_dir = normalize(make_float3(1, -1, 1));
		const material_t mat = materials[res.materialIndex];
		float angle = light_dir.x * res.faceNormal.x + light_dir.y * res.faceNormal.y + light_dir.z * res.faceNormal.z;
		angle = clamp(EPSILON, 1, angle);
		angle = 1;
		r = ((float)mat.x) * angle;
		g = ((float)mat.y) * angle;
		b = ((float)mat.z) * angle;

		// r = (res.normal.x * 0.5f + 0.5f) * 255.9;
		// g = (res.normal.y * 0.5f + 0.5f) * 255.9;
		// b = (res.normal.z * 0.5f + 0.5f) * 255.9;
		r = 255;
		g = 255;
		b = 0;
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

	int x = 500;
	int y = 600;

	const float width = X_RESOLUTION;  // pixels across
	const float height = Y_RESOLUTION;  // pixels high
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
	Model_ m;
	m.tree = tree.data();
	m.details = ModelDetails();
	m.details.span = { 1, 8 };
	float tmax = 0;
	auto res = castRay(ray, m, tmax);

	if (res.hit) {

	}
}

__host__ __device__ float3 getChildFront(const float3& pf, uint8_t slot, float scale) {
	float3 cf = pf;
	if (slot & 1) cf.x += scale;
	if (slot & 2) cf.y += scale;
	if (slot & 4) cf.z += scale;
	return cf;
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

__host__ __device__ CastResult castRay(const _3D::Ray3D& ray, const Model_ model, float& tprev) {
	CastResult result;
	result.hit = false;

	const node_t* tree = model.tree;
	const uint32_t minScale = model.details.span.x;
	const uint32_t maxScale = model.details.span.y;

	// scale of the child
	uint8_t scale = maxScale;
	// half scale of the child
	float half = scale * .5f;

	node_t parent = tree[0];
	Stack_<7> parent_stack; // allows for a maximum scale of 64

	float3 mirror = make_float3(0, 0, 0);
	uint8_t mirrorMask = 0;
	if (ray.dir_inv_.x < 0) {
		mirrorMask ^= 1; 
		mirror.x = 1;
	}
	if (ray.dir_inv_.y < 0) {
		mirrorMask ^= 2; 
		mirror.y = 1;
	}
	if (ray.dir_inv_.z < 0) {
		mirrorMask ^= 4; 
		mirror.z = 1;
	}
	float3 mirroredOrigin = mirror * scale;

	float3 inv_ray_dir = abs(ray.dir_inv_);
	float3 rayPos = ray.getPos(); // needs to mirror

	if (mirrorMask & 1) rayPos.x = 2 * mirroredOrigin.x - rayPos.x;
	if (mirrorMask & 2) rayPos.y = 2 * mirroredOrigin.y - rayPos.y;
	if (mirrorMask & 4) rayPos.z = 2 * mirroredOrigin.z - rayPos.z;

	auto tFunc = [&](const float3& a) -> const float3& {
		return (a - rayPos) * inv_ray_dir;
	};

	// mirrorMask = 0;

	float tmin = 0;
	float tmax = 0;

	float3 pmin = mirroredOrigin;
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
	if (t_pf.x < 0 && t_pb.x < 0) {
		return result;
	}
	if (t_pf.y < 0 && t_pb.y < 0) {
		return result;
	}
	if (t_pf.z < 0 && t_pb.z < 0) {
		return result;
	}


	uint8_t childSlot = 0;
	if (t_pc.x <= tmin) childSlot ^= 1;
	if (t_pc.y <= tmin) childSlot ^= 2;
	if (t_pc.z <= tmin) childSlot ^= 4;
	childSlot %= 7;

	scale = half;
	half *= 0.5f;

	bool _;
	int32_t idx = 0;
	uint8_t ittertion = 0;
	while (ittertion++ < MAX_ITTERATIONS) {
		parent = tree[idx];

		float3 cmin = getChildFront(pmin, childSlot, scale);
		float3 t_cf = tFunc(cmin);
		tmin = max(t_cf);
		float3 ccen = cmin + half;
		float3 t_cc = tFunc(ccen);

		float3 cmax = cmin + scale;
		float3 t_cb = tFunc(cmax);
		float tc_max = min(t_cb);

		uint8_t mirroredChildSlot = childSlot ^ mirrorMask;
		uint8_t validMask = parent.child_data >> 8;
		if (validMask >> mirroredChildSlot & 0x1) {
			// child is valid
			if (parent.child_data >> mirroredChildSlot & 0x1 || half < minScale) {
				// is leaf
				result.hit = true;
				tprev = tmin;
				return result;
			}


			// PUSH
			if (tc_max < tmax) {
				parent_stack.push(idx, childSlot, scale * 2);
			}
			idx = getChildIndex_(parent.child_data, mirroredChildSlot);
			parent = tree[idx];


			// the next child
			childSlot = 0;
			if (t_cc.x <= tmin) childSlot ^= 1;
			if (t_cc.y <= tmin) childSlot ^= 2;
			if (t_cc.z <= tmin) childSlot ^= 4;
			childSlot %= 7;

			pmin = cmin;
			scale = half;
			half *= .5f;
			t_pf = tFunc(pmin);
			tmin = max(t_pf);

			pmax = pmin + scale;
			float3 t_pb = tFunc(pmax);
			tmax = min(t_pb);

			continue;
			
			// get centre t
			// if less than child back
			// PUSH
			// else
			// POP or ADVANCE
		}
		// tmax = tc_max;

		uint8_t prevSlot = childSlot;
		if (t_cb.x <= tc_max) childSlot ^= 1;
		if (t_cb.y <= tc_max) childSlot ^= 2;
		if (t_cb.z <= tc_max) childSlot ^= 4;
		if (prevSlot & ~childSlot) {
			// POP
			uint32_t delta = prevSlot ^ childSlot;
			scale *= 2; // now the scale of the parent to be poped off
			auto item = parent_stack.pop(scale);
			idx = item.parentIdx;
			childSlot = item.slot ^ delta;

			if (delta & item.slot) {
				// will be called incorreclty if you need to pop more than once in a row
				return result;
			}
			half = scale * 2;

			if (item.slot & 1) pmin.x -= scale;
			if (item.slot & 2) pmin.y -= scale;
			if (item.slot & 4) pmin.z -= scale;


			t_pf = tFunc(pmin);
			tmin = max(t_pf);
		}
	}

	// if tmin < tmax then it hits
	return result;
}

// change stack to be indexed by scale, and poping will get the closest scale that is >= requested