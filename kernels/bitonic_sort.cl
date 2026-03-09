__kernel void bitonic_step(__global int* data, const uint stage, const uint step)
{
    const uint i = get_global_id(0);
    const uint j = i ^ step;

    if (j <= i) {
        return;
    }

    const int a = data[i];
    const int b = data[j];
    const int ascending = ((i & stage) == 0);

    if ((ascending && a > b) || (!ascending && a < b)) {
        data[i] = b;
        data[j] = a;
    }
}
