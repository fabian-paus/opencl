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

#define WINDOW_RADIUS 4

__kernel 
void filter_G(__read_only image2d_t firstImage,
			  __read_only image2d_t secondImage,
			  __write_only image2d_t G)
{
	const int posX = get_global_id(0);
	const int posY = get_global_id(1);

	int Ix2 = 0;
	int IxIy = 0;
	int Iy2 = 0;
	for (int y = -WINDOW_RADIUS; y <= WINDOW_RADIUS; y++) 
	{
		for (int x = -WINDOW_RADIUS; x <= WINDOW_RADIUS ; x++) 
		{
			int2 samplePos = { posX + x, posY + y };
			int ix = read_imagei(firstImage, sampler, samplePos).x;
			int iy = read_imagei(secondImage, sampler, samplePos).x;

			Ix2 += ix * ix;
			Iy2 += iy * iy;
			IxIy += ix * iy;
		}
	}

	int4 G2x2 = (int4)(Ix2, IxIy, IxIy, Iy2);
	write_imagei(G, (int2)(posX, posY), G2x2);
}

__kernel 
void scharr_x_horizontal(__read_only image2d_t source,
						 __write_only image2d_t destination)
{
	const int xPos = get_global_id(0);
    const int yPos = get_global_id(1);

    int x0 = read_imagei(source, sampler, (int2)(xPos - 1, yPos)).x;
    int x2 = read_imagei(source, sampler, (int2)(xPos + 1, yPos)).x;
    int output = x2 - x0; 

	write_imagei(destination, (int2)(xPos, yPos), (int4)(output, 0, 0, 0)); 
}

__kernel 
void scharr_x_vertical(__read_only image2d_t source,
	     			   __write_only image2d_t destination)
{
	const int xPos = get_global_id(0);
    const int yPos = get_global_id(1);

    int x0 = read_imagei(source, sampler, (int2)(xPos, yPos - 1)).x;
    int x1 = read_imagei(source, sampler, (int2)(xPos, yPos)).x;
    int x2 = read_imagei(source, sampler, (int2)(xPos, yPos + 1)).x;

    int output = 3 * x0 + 10 * x1 + 3 * x2;
	write_imagei(destination, (int2)(xPos, yPos), (int4)(output, 0, 0, 0)); 
}

__kernel 
void scharr_y_horizontal(__read_only image2d_t source,
	     			     __write_only image2d_t destination)
{
	const int xPos = get_global_id(0);
    const int yPos = get_global_id(1);

    int x0 = read_imagei(source, sampler, (int2)(xPos - 1, yPos)).x;
    int x1 = read_imagei(source, sampler, (int2)(xPos, yPos)).x;
    int x2 = read_imagei(source, sampler, (int2)(xPos + 1, yPos)).x;

    int output = 3 * x0 + 10 * x1 + 3 * x2;
	write_imagei(destination, (int2)(xPos, yPos), (int4)(output, 0, 0, 0)); 
}

__kernel 
void scharr_y_vertical(__read_only image2d_t source,
					   __write_only image2d_t destination)
{
	const int xPos = get_global_id(0);
    const int yPos = get_global_id(1);

    int x0 = read_imagei(source, sampler, (int2)(xPos, yPos - 1)).x;
    int x2 = read_imagei(source, sampler, (int2)(xPos, yPos + 1)).x;
    int output = x2 - x0; 

	write_imagei(destination, (int2)(xPos, yPos), (int4)(output, 0, 0, 0)); 
}

