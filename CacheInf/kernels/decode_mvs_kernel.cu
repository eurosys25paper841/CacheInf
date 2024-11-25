#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


const int cuda_block_dim = 16;

template <typename scalar_t>
__global__ void gather_bilinear_grid_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3> mvs,
    torch::PackedTensorAccessor32<float,3> grid,
    const int block_size, const int blocks_x, const int blocks_y){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float half_height = (block_size * blocks_x - 1) / 2.;
    float half_width = (block_size * blocks_y - 1) / 2.;
    if (x < blocks_x && y < blocks_y){
        scalar_t mvs_x = mvs[x][y][0];
        scalar_t mvs_y = mvs[x][y][1];
        for (int i = x * block_size; i < (x+1)*block_size; i++)
            for (int j = y * block_size; j < (y+1) * block_size; j++){
                // grid[i][j][0] = (j + mvs_y) / half_width - 1;    // align to [-1.,1.]
                // grid[i][j][1] = (i + mvs_x) / half_height - 1;      // align to [-1.,1.]
                grid[i][j][0] = (j + mvs_y ) / half_width - 1;    // align to [-1.,1.]
                grid[i][j][1] = (i + mvs_x) / half_height - 1;      // align to [-1.,1.]
            }
    }
}

/// @brief Grid sample from refer frame to reconstruct a new frame based on the motion vectors
/// @param refer_frame 
/// @param mvs 
/// @param block_size 
/// @return 
torch::Tensor cuda_decode_mvs(
    const torch::Tensor &refer_frame, const torch::Tensor &mvs,
    const int &block_size, const bool &bilinear){
    int dim = refer_frame.dim();
    int frame_height = refer_frame.sizes()[dim-2], frame_width = refer_frame.sizes()[dim-1];

    int channels = refer_frame.sizes()[0];
    TORCH_CHECK(dim == 3, "Expected input dim to be 3 [C,H,W], but got ", dim);
    int blocks_x = mvs.sizes()[0], blocks_y = mvs.sizes()[1];

    const auto float_options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.layout(torch::kStrided)
		.device(torch::kCUDA, 0)
		.requires_grad(false);

    auto grid = torch::zeros({frame_height, frame_width, 2}, float_options);

    dim3 kernel_blocks((blocks_x + cuda_block_dim - 1) / cuda_block_dim,
                    (blocks_y + cuda_block_dim - 1) / cuda_block_dim);
    dim3 threads_per_block(cuda_block_dim, cuda_block_dim);
    AT_DISPATCH_ALL_TYPES(mvs.scalar_type(), "gather_bilinear_grid_kernel", ([&] {
        gather_bilinear_grid_kernel<<<kernel_blocks, threads_per_block>>>(
            mvs.packed_accessor32<scalar_t, 3>(), grid.packed_accessor32<float, 3>(),
            block_size, blocks_x, blocks_y);
    }));

    if (bilinear)
        return torch::nn::functional::grid_sample(
            refer_frame.unsqueeze(0), grid.unsqueeze(0),
            torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false)).squeeze(0);
    else
        return torch::nn::functional::grid_sample(
            refer_frame.unsqueeze(0), grid.unsqueeze(0),
            torch::nn::functional::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(false)).squeeze(0);
}

