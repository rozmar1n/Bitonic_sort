__kernel void increment_kernel(__global int* data) {
    const uint id = get_global_id(0);
    data[id] += 1;
}
