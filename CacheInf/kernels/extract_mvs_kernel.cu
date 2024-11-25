#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define NDEBUG 1

int block_dim = 16;


template <typename scalar_t, typename ...T>
__global__ void cuda_block_match_kernel(
	const torch::PackedTensorAccessor32<scalar_t,3> device_ref_frame,
	const torch::PackedTensorAccessor32<scalar_t,3> device_query_frame,
	const torch::PackedTensorAccessor32<short,3> device_macroblocks,
	torch::PackedTensorAccessor32<float,3> device_MSE_all_searches,
	int device_block_height,int device_block_width,
	int device_rows, int device_cols,
	int device_channels, int device_search_dist_x,
	int device_search_dist_y)
{
	#ifndef NDEBUG
		if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
		{
			printf("IN KERNEL\n");
		}
		__syncthreads();
	#endif

	//variable to store the MSE of the macroblock - search block pair
	__shared__ float MSE_pair;
	//variable to hold the index of the macroblock
	__shared__ int block_x;
	__shared__ int block_y;

	//array to store results of the pixel_level MSE
	extern __shared__ float pixel_MSE[];
	//threads in the same block have the same search block start and stop co-ordinates
	__shared__ int search_area_x_start, search_area_x_stop, search_area_y_start, search_area_y_stop;

	pixel_MSE[threadIdx.x * device_block_width + threadIdx.y] = 0.;

	//Initial setup of variables
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		MSE_pair =0;
		block_y = (blockIdx.y/9);
		block_x = blockIdx.x;

		//%3 is needed since for 3 search blocks, the condition applies
		if (int(blockIdx.y% 3) == 0)
		{
			// divide by 9 is needed since every macroblock has 9 blocks along the width. This is not required for the height
			search_area_y_start = device_macroblocks[block_x][block_y][1] + int(blockIdx.y/9)*device_block_width - device_search_dist_y;
		}
		else if (int(blockIdx.y% 3) == 2)
		{
			search_area_y_start = device_macroblocks[block_x][block_y][1] + int(blockIdx.y/9)*device_block_width + device_search_dist_y;
		}
		else
		{
			search_area_y_start = device_macroblocks[block_x][block_y][1] + int(blockIdx.y/9)*device_block_width;
		}

		search_area_y_stop = search_area_y_start+device_block_width;

		//division by 3 and %3 is needed since, although this condition applies for 3 blocks, this time it is for the vertical
		//movement rather than the horizontal movement
		if (int((blockIdx.y/3)%3) == 0)
		{
			search_area_x_start = device_macroblocks[block_x][block_y][0] + blockIdx.x*device_block_height - device_search_dist_x;

		}
		else if (int((blockIdx.y/3)%3) == 2)
		{
			search_area_x_start = device_macroblocks[block_x][block_y][0] + blockIdx.x*device_block_height + device_search_dist_x;
		}
		else
		{
			search_area_x_start = device_macroblocks[block_x][block_y][0] + blockIdx.x*device_block_height;
		}

		search_area_x_stop = search_area_x_start+device_block_height;
	}

	__syncthreads();

	//if all of the search area parameters are within bounds
	if((search_area_x_start >= 0) && (search_area_x_stop <= device_rows) && (search_area_y_start >= 0) && (search_area_y_stop <= device_cols))
	{
		//get pixel co-ordinates for both frame 1 and frame 2
		int pixel_x_f1 = search_area_x_start+threadIdx.x;
		int pixel_y_f1 = search_area_y_start+threadIdx.y;

		int pixel_x_f2 = blockIdx.x*device_block_height+threadIdx.x;
		int pixel_y_f2 = int(blockIdx.y/9)*device_block_width+threadIdx.y;

		//get the pixel intensities and calculate the MSE
		float val = 0.;
		#ifndef NDEBUG
		int flag = 0;
		#endif
		float pixel_1;
		float pixel_2;
		for (int channel = 0; channel<device_channels; channel++)
		{
			//put the mse in the shared array
			pixel_1 = device_ref_frame[channel][pixel_x_f1][pixel_y_f1];
			pixel_2 = device_query_frame[channel][pixel_x_f2][pixel_y_f2];
			val += (pixel_1 - pixel_2) * (pixel_1 - pixel_2);
			#ifndef NDEBUG
			if (((val>50000000) || pixel_x_f1 < 0 || pixel_x_f1 >= device_rows || pixel_y_f1 < 0 || pixel_y_f1 >= device_cols ||pixel_x_f2 < 0 || pixel_x_f2 >= device_rows || pixel_y_f2 < 0 || pixel_y_f2 >= device_cols || (std::isnan(val))) && (flag < 1)){
				printf("Error! val %.4f blockIdx [%d,%d] threadIdx [%d,%d] pixel: f1 [%d,%d] f2 [%d,%d] pixel1 %.4f pixel2 %.4f\n", val, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, pixel_x_f1, pixel_y_f1, pixel_x_f2, pixel_y_f2, pixel_1, pixel_2);
				flag = 1;
			}
			#endif
		}
		pixel_MSE[threadIdx.y+threadIdx.x*device_block_width] = val;
	}

	__syncthreads();

	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		//if within the search area, calculate the mse for the macroblock search block pair
		if((search_area_x_start >= 0) && (search_area_x_stop <= device_rows) && (search_area_y_start >= 0) && (search_area_y_stop <= device_cols))
		{
			for (int row = 0; row<device_block_height; row++)
			{
				for (int col = 0; col<device_block_width; col++)
				{
					MSE_pair = MSE_pair +pixel_MSE[col+row*device_block_width];
				}
			}
			MSE_pair = MSE_pair/float(device_block_height*device_block_width);
			//write the MSE pair to a variable which will hold all the MSEs for all the macroblock and search block permutations
			device_MSE_all_searches[block_x][block_y][int(blockIdx.y%9)] = MSE_pair;
			#ifndef NDEBUG
				printf("Block idx [%d,%d] searched block starting at [%d,%d], mse %f\n", block_x, block_y, search_area_x_start, search_area_y_start, MSE_pair);
			#endif
		}
	}

	#ifndef NDEBUG
		__syncthreads();
		if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
		{
			printf("Finished Kernel Execution\n");
		}
	#endif
}

__global__ void cuda_result_update_kernel(
	torch::PackedTensorAccessor32<short,3> device_macroblocks,
	torch::PackedTensorAccessor32<float,2> device_macroblocks_mse,
	const torch::PackedTensorAccessor32<float,3> device_MSE_all_searches,
 	int search_dist_x, int search_dist_y,
	int blocks_x, int blocks_y
){
	#ifndef NDEBUG
		if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
		{
			printf("Updating\n");
		}
		__syncthreads();
	#endif
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	#ifndef NDEBUG
	int flag = 0;
	#endif
	if ((x < blocks_x) && (y < blocks_y)){
		int motion_vector_x_this_search = 0;
		int motion_vector_y_this_search = 0;
		//check all the possible search block pairs
		for (int _search_block = 0 ; _search_block<9; _search_block++)
		{
			//if the mse for the macroblock search block pair is less that the currently set
			//mse, update the parameters for that macroblock
			//NB: MSE_all searches = 9*number of macroblocks, where:
				//index%9 = 0 -> top left block
				//index%9 = 1 -> top centre block
				//index%9 = 2 -> top right block
				//index%9 = 3 -> middle left block
				//index%9 = 4 -> middle centre block
				//index%9 = 5 -> middle right left block
				//index%9 = 6 -> bottom left block
				//index%9 = 7 -> bottom centre block
				//index%9 = 8 -> bottom right block
			auto search_block = _search_block % 9;

			float mse_all_search = device_MSE_all_searches[x][y][search_block];
			if(mse_all_search < device_macroblocks_mse[x][y])
			{
				device_macroblocks_mse[x][y] = mse_all_search;
				motion_vector_y_this_search = ((search_block%3)-1)*search_dist_y;
				motion_vector_x_this_search = ((search_block/3)-1)*search_dist_x;
				#ifndef NDEBUG
				flag = 1;
				#endif
			}
		}
		#ifndef NDEBUG
		if (flag==1){
			printf("block [%d,%d] mvs [%d,%d] act [%d,%d]\n", x,y,device_macroblocks[x][y][0],device_macroblocks[x][y][1],motion_vector_x_this_search,motion_vector_y_this_search);
		}
		#endif

		device_macroblocks[x][y][0] =
			device_macroblocks[x][y][0] + motion_vector_x_this_search;
		device_macroblocks[x][y][1] =
			device_macroblocks[x][y][1] + motion_vector_y_this_search;
	}
	#ifndef NDEBUG
		__syncthreads();
		if((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
		{
			printf("Finish updating\n");
		}
	#endif
}


std::vector<torch::Tensor> cuda_extract_mvs(
    const torch::Tensor &ref_frame, const torch::Tensor &query_frame, const int &block_size) {
	const int dims = ref_frame.dim();
	//1st index = blocks along x, 2nd index => blocks along y
	//This 2D array will contain motion vectors in the x and y directions and MSE for each macroblock in the image
	const int columns = ref_frame.sizes()[dims-1];
	const int rows = ref_frame.sizes()[dims-2];
	const int channels = ref_frame.sizes()[dims-3];

	//Parameters for the algorithm: macroblock width and height and search area parameters
	const int block_width = block_size;
	const int block_height = block_size;
	const int search_vertical = block_size;
	const int search_horizontal = block_size;
	//the number of macroblocks along the x and y directions
	const int blocks_y = columns/block_width;
	const int blocks_x = rows/block_height;

	int search_dist_y = block_size * 4;
	int search_dist_x = block_size * 4;
	// int search_dist_y = 32;
	// int search_dist_x = 32;
	const int time = std::sqrt(search_dist_y);

	const int blocks_update_y = (blocks_y +  block_dim - 1) / block_dim;
	const int blocks_update_x = (blocks_x +  block_dim - 1) / block_dim;

	// Using short integer to represent motion vectors following the standard h264 motion vectors.
	const auto short_options = torch::TensorOptions()
		.dtype(torch::kInt16)
		.layout(torch::kStrided)
		.device(torch::kCUDA, 0)
		.requires_grad(false);

	const auto float_options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.layout(torch::kStrided)
		.device(torch::kCUDA, 0)
		.requires_grad(false);

	// torch::Tensor result = torch::empty({blocks_x*blocks_y}, options);

	//Initialisation of the parmeters for each block
	torch::Tensor device_macroblocks = torch::zeros({blocks_x, blocks_y,  2}, short_options);

	torch::Tensor device_macroblocks_mse = torch::full({blocks_x, blocks_y}, std::numeric_limits<float>::max(), float_options);
	// device_macroblocks_mse.index_put_({"..."}, std::numeric_limits<float>::max());

	torch::Tensor device_MSE_all_searches = torch::full({blocks_x, blocks_y, 9}, std::numeric_limits<float>::max(), float_options);

	// auto device_ref_frame = ref_frame;
	// auto device_query_frame = query_frame;

	const dim3 Match_Kernel_Blocks(blocks_x, 9*blocks_y);
	const dim3 Match_Threads_Per_Block(block_height, block_width);
	const dim3 Update_Kernel_Blocks(blocks_update_x, blocks_update_y);
	const dim3 Update_Threads_Per_Block(block_dim, block_dim);
	for (int search_count = 0; search_count<time + 2; search_count++)	//for loop to denote the step in which the 3 step search has reached
	{
		//call the kernel

		AT_DISPATCH_ALL_TYPES(ref_frame.scalar_type(), "cuda_extract_mvs", ([&] {
			cuda_block_match_kernel<<<Match_Kernel_Blocks,Match_Threads_Per_Block, sizeof(float)*block_width*block_height>>>(
			ref_frame.packed_accessor32<scalar_t, 3>(), query_frame.packed_accessor32<scalar_t, 3>(),
			device_macroblocks.packed_accessor32<short, 3>(),
			device_MSE_all_searches.packed_accessor32<float, 3>(),
			block_height, block_width, rows, columns, channels, search_dist_x, search_dist_y);
		}));


		cuda_result_update_kernel<<<Update_Kernel_Blocks, Update_Threads_Per_Block>>>(
			device_macroblocks.packed_accessor32<short, 3>(),
			device_macroblocks_mse.packed_accessor32<float, 2>(),
			device_MSE_all_searches.packed_accessor32<float, 3>(),
			search_dist_x, search_dist_y, blocks_x, blocks_y);

		//Update also the search dist parameters to get finer searches.
		//After 3 iterations, they should be set to 1 such that macroblocks differ by 1 pixel
		if(search_count >= (time - 2))
		{
			search_dist_x = 1;
			search_dist_y = 1;
		}
		else
		{
			//using fast ceil - must be done since no guarantee division will result in exact multiples
			search_dist_x = int((search_dist_x+(search_dist_x/2)-1)/((search_dist_x/2)));
			search_dist_y = int((search_dist_y+(search_dist_y/2)-1)/((search_dist_y/2)));
		}
	}
  	return {device_macroblocks, device_macroblocks_mse};
}
