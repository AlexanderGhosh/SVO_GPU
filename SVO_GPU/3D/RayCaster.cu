#include "RayCaster.cuh"

using namespace _3D;

RayCastResult3D::RayCastResult3D() : hit(false), pos(), normal(), material_index(0)
{
}

slot_t _3D::push(const float3& t_front, const float3& t_centre)
{
	// return 0;
	// const float t_f_max = max(t_front);
	// const float t_c_min = min(t_centre);

	// if (t_f_max <= t_c_min) return 0;
	// slot_t res = 0;
	// // if it get here then at least 1 of t_centre < t_front the ones which are you advance (not ADVANCE) the slot in that direction
	// // so where t_centre == t_c_min advance (not ADVANCE) along
	// if (t_centre.x == t_c_min) res += 1;
	// if (t_centre.y == t_c_min) res += 2;
	// if (t_centre.z == t_c_min) res += 4;

	// /*if (t_centre.x <= t_c_min) res += 1;
	// if (t_centre.y <= t_c_min) res += 2;
	// if (t_centre.z <= t_c_min) res += 4;*/
	// return res;

	/*
	if x front < x centre [0, 2, 4, 6]
	if y front < y centre [0, 1, 4, 5]
	if z front < z centre [0, 1, 2, 3]
	*/
	slot_t slot = 7;
	const float t_f_max = max(t_front);
	if (t_f_max < t_centre.x) slot -= 1;
	if (t_f_max < t_centre.y) slot -= 2;
	if (t_f_max < t_centre.z) slot -= 4;
	return slot;
}

slot_t _3D::advance(slot_t slot, const float3& t_back, bool& should_pop)
{
	float t_b_min = fminf(t_back.x, fminf(t_back.y, t_back.z));
	t_b_min = min(t_back);
	should_pop = false;
	if (t_back.x == t_b_min) {
		slot++;
		should_pop = slot == 2;
		should_pop |= slot == 4;
		should_pop |= slot == 6;
		should_pop |= slot == 8;
	}
	if (t_back.y == t_b_min) {
		switch (slot) {
		case 2:
		case 3:
		case 6:
		case 7:
			should_pop = true;
			break;
		default:
			slot += 2;
			break;
		}
	}
	if (t_back.z == t_b_min) {
		slot += 4;
		should_pop |= slot > 7;
	}
	return slot;
}

float3 _3D::getChildPosition(float3 parent_front, const slot_t next_slot, const float next_scale)
{
	if (next_slot & 1) parent_front.x += next_scale;
	if (next_slot & 2) parent_front.y += next_scale;
	if (next_slot & 4) parent_front.z += next_scale;
	return parent_front;
}

slot_t _3D::mirrorSlot(slot_t slot, const mirror_t& mirror)
{
	if (mirror.x == -1) {
		if (slot % 2 == 0) {
			slot++;
		}
		else {
			slot--;
		}
	}
	if (mirror.y == -1) {
		switch (slot) {
		case 0:
		case 1:
		case 4:
		case 5:
			slot += 2;
			break;
		case 2:
		case 3:
		case 6:
		case 7:
			slot -= 2;
			break;
		}
	}
	if (mirror.z == -1) {
		if (slot > 3) {
			slot -= 4;
		}
		else {
			slot += 4;
		}
	}
	return slot;
}

float3 _3D::getMirrorLine(const float3& orign, float scale, const mirror_t& mirror)
{
	return orign - mirror * scale;
}

float3 _3D::mirrorAround(float3 pos, const float3& mirror_line, const mirror_t& mirror)
{
	pos -= mirror_line;
	pos -= mirror;
	pos = mirror_line - pos;
	return abs(pos);
}

int _3D::getChildIndex(const uint32_t parent, const slot_t child_slot, bool& is_leaf)
{
	is_leaf = false;
	const int child_idx = (parent & 0xffff0000) >> 16;
	const uint32_t leaf_mask = 1 << child_slot;
	const uint32_t valid_mask = leaf_mask << 8;
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
		is_leaf = parent & leaf_mask;
		return child_idx + offset;
	}
	return -1;
}

int _3D::getChildIndex(const uint32_t parent, slot_t child_slot, const mirror_t& mirror, bool& is_leaf)
{
	child_slot = mirrorSlot(child_slot, mirror);
	return getChildIndex(parent, child_slot, is_leaf);
}

Ray3D _3D::correctRay(const Ray3D& ray, const float3& mirror_line, const mirror_t& mirror)
{
	float3 p = ray.getPos();
	float3 d = ray.getDirection();
	d = abs(d);

	if (mirror.x < 0) p.x = 2 * mirror_line.x - p.x;
	if (mirror.y < 0) p.y = 2 * mirror_line.y - p.y;
	if (mirror.z < 0) p.z = 2 * mirror_line.z - p.z;

	return Ray3D(p, d);
}

bool _3D::hits_at_all(const Ray3D& ray, const float3 front, const float3 back) {
	float tmin = 0.0, tmax = FLT_MAX;
	const float3 p = ray.getPos();
	const float3 d = ray.getDirection();


	float t1 = (front.x - p.x) / d.x;
	float t2 = (back.x - p.x) / d.x;

	tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
	tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));


	t1 = (front.y - p.y) / d.y;
	t2 = (back.y - p.y) / d.y;

	tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
	tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));


	t1 = (front.z - p.z) / d.z;
	t2 = (back.z - p.z) / d.z;

	tmin = fminf(fmaxf(t1, tmin), fmaxf(t2, tmin));
	tmax = fmaxf(fminf(t1, tmax), fminf(t2, tmax));

	return tmin <= tmax;
}

uint8_t _3D::getShaderIndex(const node_t& parent, const slot_t child_slot, const mirror_t& mirror)
{
	uint32_t shift = mirrorSlot(child_slot, mirror) * 4;
	return (parent.shader_data >> shift) & 0b1111;
}

void _3D::getHitDetails(RayCastResult3D& result, const float3& t_front, const Ray3D& ray, const mirror_t& mirror)
{
	float m = max(t_front);
	result.pos = ray.point(m);
	uint32_t i = elemMax(t_front);
	result.normal = make_float3(0, 0, 0);
	if (i == 0) result.normal.x = mirror.x < 0 ? -1 : 1;
	if (i == 1) result.normal.y = mirror.y < 0 ? -1 : 1;
	if (i == 2) result.normal.z = mirror.z < 0 ? -1 : 1;
}

RayCastResult3D _3D::castRay(const Ray3D& ray, const node_t* tree)
{
	RayCastResult3D result;
	result.hit = false;

	node_t parent = tree[0];
	float parentScale = MAX_SCALE;
	float childScale = parentScale * .5f;

	// DONT CHANGE
	float3 mirror = sign(ray.getDirection());
	mirror -= 1.f;
	mirror *= .5f;

	float3 parentFront = getMirrorLine(make_float3(0, 0, 0), parentScale, mirror);
	float3 parentCentre = parentFront + childScale;
	float3 parentBack = parentCentre + childScale;

	const float3 mirrorLine = parentFront;
	const Ray3D ray_ = correctRay(ray, mirrorLine, mirror);

	if (!hits_at_all(ray_, parentFront, parentBack)) {
		return result;
	}


	auto tFunc =
		[&](const float3& a) {
			return ray_.t(a);
		};

	float3 t_pf = tFunc(parentFront);
	float3 t_pc = tFunc(parentCentre);
	float3 t_pb = tFunc(parentBack);

	if (t_pf.x < 0 && t_pb.x < 0) {
		return result;
	}
	if (t_pf.y < 0 && t_pb.y < 0) {
		return result;
	}
	if (t_pf.z < 0 && t_pb.z < 0) {
		return result;
	}


	slot_t childSlot = push(t_pf, t_pc);
	slot_t parentSlot = -1;

	bool isLeaf;
	int childIdx = getChildIndex(parent.child_data, childSlot, mirror, isLeaf);
	float3 childFront = getChildPosition(parentFront, childSlot, childScale);
	float3 t_cf = tFunc(childFront);
	if (isLeaf) {
		result.hit = true;
		// result.pos = childFront;
		result.material_index = getShaderIndex(parent, childSlot, mirror);
		// result.pos = mirrorAround(result.pos, mirrorLine, mirror);
		getHitDetails(result, t_cf, ray, mirror);
		return result;
	}


	int itteration = 0;

	Stack parentStack;
	bool justPopped = false;

	node_t child;
	float3 childCentre, childBack;
	float3 t_cc, t_cb;
	float3 nextChildPos = make_float3(0, 0, 0);
	slot_t nextChildSlot;
	float nextChildScale = childScale;
	node_t newParent;
	slot_t newParentSlot;
	//int nextIdx;
	bool shouldPop;
	while (itteration++ < MAX_ITTERATIONS) {

		if (childIdx == -1 || justPopped) {
			childBack = childFront + childScale;
			t_cb = tFunc(childBack);
			nextChildSlot = advance(childSlot, t_cb, shouldPop);
			if (!shouldPop) {
				// ADVANCE
				childIdx = getChildIndex(parent.child_data, nextChildSlot, mirror, isLeaf);
				if (isLeaf) {
					result.hit = true;
					// result.pos = getChildPosition(parentFront, nextChildSlot, nextChildScale);
					result.material_index = getShaderIndex(parent, nextChildSlot, mirror);
					// result.pos = mirrorAround(result.pos, mirrorLine, mirror);
					getHitDetails(result, tFunc(childFront), ray, mirror);
					return result;
				}

				nextChildScale = childScale;
				justPopped = false;
			}
			else {
				// POP
				if (parentScale * 2.f > MAX_SCALE) return result;

				auto t = parentStack.pop();
				newParent = t.parent;
				newParentSlot = t.slot;

				// nextChildSlot = pop(parentSlot, t_cb);
				nextChildSlot = parentSlot;

				if (parentSlot & 1) parentFront.x -= parentScale;
				if (parentSlot & 2) parentFront.y -= parentScale;
				if (parentSlot & 4) parentFront.z -= parentScale;

				// justPopped = nextChildSlot <= parentSlot;
				justPopped = true;

				nextChildScale = parentScale;
				parentScale *= 2.f;

				parent = newParent;
				parentSlot = newParentSlot;

				bool _;
				childIdx = getChildIndex(parent.child_data, nextChildSlot, mirror, _);
			}
		}
		else {
			// PUSH
			child = tree[childIdx];
			parentStack.push(parent, parentSlot);

			childCentre = childFront + childScale * .5f;
			t_cf = tFunc(childFront);
			t_cc = tFunc(childCentre);

			nextChildSlot = push(t_cf, t_cc);

			childIdx = getChildIndex(child.child_data, nextChildSlot, mirror, isLeaf);

			nextChildScale = childScale * .5f;
			if (nextChildScale < MIN_SCALE) {
				result.hit = true;
				// result.pos = childFront;
				result.material_index = getShaderIndex(child, nextChildSlot, mirror);
				// result.pos = mirrorAround(result.pos, mirrorLine, mirror);
				getHitDetails(result, t_cf, ray, mirror);
				return result;
			}
			parentFront = childFront;

			if (isLeaf) {
				result.hit = true;
				result.pos = getChildPosition(parentFront, nextChildSlot, nextChildScale);
				result.material_index = getShaderIndex(child, nextChildSlot, mirror);
				// result.pos = mirrorAround(result.pos, mirrorLine, mirror);
				getHitDetails(result, tFunc(result.pos), ray, mirror);
				return result;
			}

			parentScale = childScale;
			parent = child;
			parentSlot = childSlot;

			justPopped = false;
		}

		nextChildPos = getChildPosition(parentFront, nextChildSlot, nextChildScale);
		childScale = nextChildScale;

		childFront = nextChildPos;
		childSlot = nextChildSlot;
	}

	return result;

}
