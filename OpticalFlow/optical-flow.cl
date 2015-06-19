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

	float x0 = read_imageui(source, sampler, (int2)(ix-2,iy)).x * 0.0625f;
	float x1 = read_imageui(source, sampler, (int2)(ix-1,iy)).x * 0.25f;
	float x2 = read_imageui(source, sampler, pos).x * 0.375f;
	float x3 = read_imageui(source, sampler, (int2)(ix+1,iy)).x * 0.25f;
	float x4 = read_imageui(source, sampler, (int2)(ix+2,iy)).x * 0.0625f;

	int output = round(x0 + x1 + x2 + x3 + x4);

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

	float x0 = read_imageui(source, sampler, (int2)(ix2, iy2-2)).x * 0.0625;
	float x1 = read_imageui(source, sampler, (int2)(ix2, iy2-1)).x * 0.25f;
	float x2 = read_imageui(source, sampler, (int2)(ix2, iy2+0)).x * 0.375f;
	float x3 = read_imageui(source, sampler, (int2)(ix2, iy2+1)).x * 0.25f;
	float x4 = read_imageui(source, sampler, (int2)(ix2, iy2+2)).x * 0.0625;

	int output = round(x0 + x1 + x2 + x3 + x4);

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

#define FRAD 4
#define eps 0.0000001f;
#define LOCAL_X 16
#define LOCAL_Y 8

__kernel void optical_flow( 
    __read_only image2d_t I,
    __read_only image2d_t Ix,
    __read_only image2d_t Iy,
    __read_only image2d_t G,
    __read_only image2d_t J,
	int use_guess,
    __read_only image2d_t guess_in,
    __write_only image2d_t guess_out,
    int guess_width,
	int guess_height )
{
    // Create sampler objects.  One is for nearest neighbour, the other for
    // bilinear interpolation
    sampler_t bilinSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_LINEAR ;
    sampler_t nnSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_NEAREST ;

    // Image indices. Note for the texture, we offset by 0.5 to use the center
    // of the texel. 
    int2 iIidx = { get_global_id(0), get_global_id(1)};
    float2 Iidx = { get_global_id(0)+0.5, get_global_id(1)+0.5 };

	if (iIidx.x >= guess_width || iIidx.y >= guess_height)
	{ 
		return;
	}

    float2 g = {0,0}; 

        // Previous pyramid levels provide input guess.  Use if available.
    if (use_guess != 0)
	{
        //lookup in higher level, div by two to find position because its smaller
        int2 gin_pos = { iIidx.x/2, iIidx.y/2 };
        float2 g_in = read_imagef(guess_in, nnSampler, gin_pos).xy;
        // multiply the motion by two because we are in a larger level. 
        g.x = g_in.x * 2;
        g.y = g_in.y * 2;
    }

    float2 v = {0,0};
    
    // invert G, 2x2 matrix , use float since int32 will overflow quickly
    int4 Gmat = read_imagei(G, nnSampler, iIidx);
    float det_G = (float)Gmat.s0 * (float)Gmat.s3 - (float)Gmat.s1 * (float)Gmat.s2 ;
        // avoid possible 0 in denominator
    if (det_G == 0.0f) 
	    det_G = eps;

    float4 Ginv = { Gmat.s3/det_G, -Gmat.s1/det_G, -Gmat.s2/det_G, Gmat.s0/det_G };

    // for large motions we can approximate them faster by applying gain to the motion
    float gain = 4.f;
    for (int k = 0; k < 8; k++)
	{
        float2 Jidx = { Iidx.x + g.x + v.x, Iidx.y + g.y + v.y };
        float2 b = {0,0};
        float2 n = {0,0};

        // calculate the mismatch vector
        for (int j = -FRAD; j <= FRAD; j++) 
		{
            for (int i = - FRAD; i <= FRAD; i++) 
			{
                // this should use shared memory instead...
                int Isample = read_imageui(I, nnSampler, Iidx+(float2)(i,j)).x;
                int Jsample = read_imageui(J, bilinSampler, Jidx+(float2)(i,j)).x;
                float dIk = (float)Isample - Jsample;

                int ix = read_imagei(Ix, nnSampler, Iidx + (float2)(i,j)).x; 
                int iy = read_imagei(Iy, nnSampler, Iidx + (float2)(i,j)).x; 

                b += (float2)(dIk * ix * gain, dIk * iy * gain);
            }
        }

        // Optical flow (Lucas-Kanade).  
        //  Solve n = G^-1 * b
        //compute n (update), mult Ginv matrix by vector b
        n = (float2)(Ginv.s0*b.s0 + Ginv.s1*b.s1,  Ginv.s2*b.s0 + Ginv.s3*b.s1);

        // if the determinant is not plausible, suppress motion at this pixel
        if (fabs(det_G) < 1000) 
			n = (float2)(0,0);

        // break if no motion
        // on test images this changes from 74 ms if no break, 55 if break, on minicooper, k=8, FRAD=4, gain=4
        if (length(n) < 0.004) 
			break;

        // guess for next iteration: v_new = v_current + n
        v = v + n;
    }

    int2 outCoords = { get_global_id(0), get_global_id(1) }; 

    write_imagef(guess_out, outCoords, (float4)(v.x + g.x, v.y + g.y, 0.0f, 0.0f));
}

__kernel void optical_flow_2( 
    __read_only image2d_t I,
    __read_only image2d_t Ix,
    __read_only image2d_t Iy,
    __read_only image2d_t G,
    __read_only image2d_t J,
	int use_guess,
    __read_only image2d_t guess_in,
    __write_only image2d_t guess_out,
    int guess_width,
	int guess_height )
{
    // Create sampler objects.  One is for nearest neighbour, the other fo
    // bilinear interpolation
    sampler_t bilinSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_LINEAR ;
    sampler_t nnSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_NEAREST ;

						   
    // declare some shared memory
    __local int smem[2*FRAD + LOCAL_Y][2*FRAD + LOCAL_X] ;
    __local int smemIy[2*FRAD + LOCAL_Y][2*FRAD + LOCAL_X] ;
    __local int smemIx[2*FRAD + LOCAL_Y][2*FRAD + LOCAL_X] ;

    // Image indices. Note for the texture, we offset by 0.5 to use the centre
    // of the texel. 
    int2 iIidx = { get_global_id(0), get_global_id(1)};
    float2 Iidx = { get_global_id(0)+0.5, get_global_id(1)+0.5 };

	// load some data into local memory because it will be re-used frequently
    // load upper left region of smem
    int2 tIdx = { get_local_id(0), get_local_id(1) }; // 0..15, 0..7
    smem[ tIdx.y ][ tIdx.x ] = read_imageui( I, nnSampler, Iidx+(float2)(-FRAD,-FRAD) ).x;
    smemIy[ tIdx.y ][ tIdx.x ] = read_imageui( Iy, nnSampler, Iidx+(float2)(-FRAD,-FRAD) ).x;
    smemIx[ tIdx.y ][ tIdx.x ] = read_imageui( Ix, nnSampler, Iidx+(float2)(-FRAD,-FRAD) ).x;

    // upper right
    if( tIdx.x < 2*FRAD ) { 
            smem[ tIdx.y ][ tIdx.x + LOCAL_X ] = read_imageui( I, nnSampler, Iidx+(float2)(LOCAL_X - FRAD,-FRAD) ).x;
            smemIy[ tIdx.y ][ tIdx.x + LOCAL_X ] = read_imageui( Iy, nnSampler, Iidx+(float2)(LOCAL_X - FRAD,-FRAD) ).x;
            smemIx[ tIdx.y ][ tIdx.x + LOCAL_X ] = read_imageui( Ix, nnSampler, Iidx+(float2)(LOCAL_X - FRAD,-FRAD) ).x;
    }
    // lower left
    if( tIdx.y < 2*FRAD ) {
            smem[ tIdx.y + LOCAL_Y ][ tIdx.x ] = read_imageui( I, nnSampler, Iidx+(float2)(-FRAD, LOCAL_Y-FRAD) ).x;
            smemIy[ tIdx.y + LOCAL_Y ][ tIdx.x ] = read_imageui( Iy, nnSampler, Iidx+(float2)(-FRAD, LOCAL_Y-FRAD) ).x;
            smemIx[ tIdx.y + LOCAL_Y ][ tIdx.x ] = read_imageui( Ix, nnSampler, Iidx+(float2)(-FRAD, LOCAL_Y-FRAD) ).x;
    }
    // lower right
    if( tIdx.x < 2*FRAD && tIdx.y < 2*FRAD ) {
            smem[ tIdx.y + LOCAL_Y ][ tIdx.x + LOCAL_X ] = read_imageui( I, nnSampler, Iidx+(float2)(LOCAL_X - FRAD, LOCAL_Y - FRAD) ).x;
            smemIy[ tIdx.y + LOCAL_Y ][ tIdx.x + LOCAL_X ] = read_imageui( Iy, nnSampler, Iidx+(float2)(LOCAL_X - FRAD, LOCAL_Y - FRAD) ).x;
            smemIx[ tIdx.y + LOCAL_Y ][ tIdx.x + LOCAL_X ] = read_imageui( Ix, nnSampler, Iidx+(float2)(LOCAL_X - FRAD, LOCAL_Y - FRAD) ).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	if (iIidx.x >= guess_width || iIidx.y >= guess_height)
	{ 
		return;
	}

    float2 g = {0,0}; 

        // Previous pyramid levels provide input guess.  Use if available.
    if (use_guess != 0)
	{
        //lookup in higher level, div by two to find position because its smaller
        int2 gin_pos = { iIidx.x/2, iIidx.y/2 };
        float2 g_in = read_imagef(guess_in, nnSampler, gin_pos).xy;
        // multiply the motion by two because we are in a larger level. 
        g.x = g_in.x * 2;
        g.y = g_in.y * 2;
    }

    float2 v = {0,0};
    
    // invert G, 2x2 matrix , use float since int32 will overflow quickly
    int4 Gmat = read_imagei(G, nnSampler, iIidx);
    float det_G = (float)Gmat.s0 * (float)Gmat.s3 - (float)Gmat.s1 * (float)Gmat.s2 ;
        // avoid possible 0 in denominator
    if (det_G == 0.0f) 
	    det_G = eps;

    float4 Ginv = { Gmat.s3/det_G, -Gmat.s1/det_G, -Gmat.s2/det_G, Gmat.s0/det_G };

    // for large motions we can approximate them faster by applying gain to the motion
    float gain = 4.f;
    for (int k=0 ; k < 8 ; k++)
	{
        float2 Jidx = { Iidx.x + g.x + v.x, Iidx.y + g.y + v.y };
        float2 b = {0,0};
        float2 n = {0,0};

        // calculate the mismatch vector
        for (int j = -FRAD; j <= FRAD; j++) 
		{
            for (int i = - FRAD; i <= FRAD; i++) 
			{
                // this should use shared memory instead...
                int Isample = smem[tIdx.y + FRAD +j][tIdx.x + FRAD+ i];
                int Jsample = read_imageui(J, bilinSampler, Jidx+(float2)(i,j)).x;
                float dIk = (float)Isample - Jsample;

                int ix = smemIx[tIdx.y + FRAD +j][tIdx.x + FRAD+ i]; 
                int iy = smemIy[tIdx.y + FRAD +j][tIdx.x + FRAD+ i]; 

                b += (float2)(dIk * ix * gain, dIk * iy * gain);
            }
        }

        // Optical flow (Lucas-Kanade).  
        //  Solve n = G^-1 * b
        //compute n (update), mult Ginv matrix by vector b
        n = (float2)(Ginv.s0*b.s0 + Ginv.s1*b.s1,  Ginv.s2*b.s0 + Ginv.s3*b.s1);

        // if the determinant is not plausible, suppress motion at this pixel
        if (fabs(det_G) < 1000) 
			n = (float2)(0,0);

        // break if no motion
        // on test images this changes from 74 ms if no break, 55 if break, on minicooper, k=8, FRAD=4, gain=4
        if (length(n) < 0.004) 
			break;

        // guess for next iteration: v_new = v_current + n
        v = v + n;
    }

    int2 outCoords = { get_global_id(0), get_global_id(1) }; 

    write_imagef(guess_out, outCoords, (float4)(v.x + g.x, v.y + g.y, 0.0f, 0.0f));
}

