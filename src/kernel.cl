__kernel void convolution(
   	const __global float* input, __global float4* ouput, __constant float* filter_mem,
   	const int imageHeight, const int imageWidth, const int half_filter) {

	int gid = get_global_id(0);
	int idx = gid * 4;

	int row = idx / imageWidth;
	int col = idx % imageWidth;

	float4 ans = (0.0, 0.0, 0.0, 0.0); //4 float
    float4 tmp, fil;

	int i, j, filter_idx = 0;
	int cur_x, cur_y, position;
	for (i = -half_filter; i <= half_filter; ++i) {

		cur_x = row + i;
		
		for (j = -half_filter; j <= half_filter; ++j, ++filter_idx) {
			
			cur_y = col + j;
            if(cur_y>=0 && cur_y<imageWidth && cur_x>=0 && cur_x<imageHeight){
                position = (cur_x * imageWidth) + cur_y;
			
                tmp = (float4)(input[position], input[position+1], input[position+2], input[position+3]);
                fil = filter_mem[filter_idx];

                ans += tmp * fil;
            }
		}
	}
	ouput[gid] = ans;
}
