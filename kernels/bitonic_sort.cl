__kernel void noop_kernel(__global float* data) {
    const uint id = get_global_id(0);
    data[id] = data[id];
}
