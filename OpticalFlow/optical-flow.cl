const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
							CLK_ADDRESS_CLAMP_TO_EDGE |
							CLK_FILTER_NEAREST;

__kernel
void downfilter_x(__read_only image2d_t source,
                  __write_only image2d_t destination)
{
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);
	const int2 pos = { ix, iy };

	float x0 = read_imageui(source, sampler, (int2)(ix-2,iy)).x * 0.125f;
	float x1 = read_imageui(source, sampler, (int2)(ix-1,iy)).x * 0.25f;
	float x2 = read_imageui(source, sampler, pos).x * 0.375f;
	float x3 = read_imageui(source, sampler, (int2)(ix+1,iy)).x * 0.25f;
	float x4 = read_imageui(source, sampler, (int2)(ix+2,iy)).x * 0.125f;

	int output = round(x2 + (x3 + x4 + (x0 + x4)));

	write_imageui(destination, pos, (uint4)(output, 0, 0, 0));
}

__kernel
void downfilter_y(__read_only image2d_t source,
                  __write_only image2d_t destination)
{
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);
	const int ix2 = 2 * ix;
	const int iy2 = 2 * iy;

	float x0 = read_imageui(source, sampler, (int2)(ix2, iy2-2)).x * 0.125f;
	float x1 = read_imageui(source, sampler, (int2)(ix2, iy2-1)).x * 0.25f;
	float x2 = read_imageui(source, sampler, (int2)(ix2, iy2+0)).x * 0.375f;
	float x3 = read_imageui(source, sampler, (int2)(ix2, iy2+1)).x * 0.25f;
	float x4 = read_imageui(source, sampler, (int2)(ix2, iy2+2)).x * 0.125f;

	int output = round(x2 + (x3 + x4 + (x0 + x4)));

	write_imageui(destination, (int2)(ix, iy), (uint4)(output, 0, 0, 0));
}