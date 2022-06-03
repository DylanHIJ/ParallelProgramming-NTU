#define uint32_t unsigned int

inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}
inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void vecDot (
    uint32_t key1,
    uint32_t key2,
    __global uint32_t *bufferVec
) {
    __local uint32_t buf[1024];

    int globalID = get_global_id(0),
        groupID = get_group_id(0),
        localID = get_local_id(0),
        localSize = get_local_size(0);

    buf[localID] = encrypt(globalID, key1) * encrypt(globalID, key2);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localSize / 2; i > 0; i /= 2) {
        if (localID < i)
            buf[localID] += buf[localID+i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localID == 0)
        bufferVec[groupID] = buf[0];
}