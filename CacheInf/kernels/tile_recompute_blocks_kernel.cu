#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int thread_dim_h = 16;
const int thread_dim_w = 32;
template <typename scalar_t, typename ...T>
__global__ void cuda_tile_recompute_blocks_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3> frame,
	const torch::PackedTensorAccessor32<short,2> activation_indices,
	torch::PackedTensorAccessor32<scalar_t,4> tiled_receptive_field,
    const int mvs_block_size, const int offset, const int stride,
    const int frame_height, const int frame_width,
    const int blocks_thread_x, const int blocks_thread_y){
    int channel = blockIdx.x;
    int activation_idx = blockIdx.y;
    int a_x = activation_indices[activation_idx][0];
    int a_y = activation_indices[activation_idx][1];
    int bottom_x = a_x * stride - offset;
    int bottom_y = a_y * stride - offset;
    int field_offset_x = threadIdx.x * blocks_thread_x * mvs_block_size;
    int field_offset_y = threadIdx.y * blocks_thread_y * mvs_block_size;

    auto sub_tiled_receptive_field = tiled_receptive_field[activation_idx][channel];
    auto sub_frame = frame[channel];

    int _field_offset_x;
    int _field_offset_y;
    int _frame_offset_x;
    int _frame_offset_y;
    for (int i=0; i < mvs_block_size * blocks_thread_x; i++){
        _field_offset_x = field_offset_x + i;
        _frame_offset_x = _field_offset_x + bottom_x;
        if (_frame_offset_x >= 0 && _frame_offset_x < frame_height){
            for (int j=0; j < mvs_block_size * blocks_thread_y; j++){
                _field_offset_y = field_offset_y + j;
                _frame_offset_y = _field_offset_y + bottom_y;
                if (_frame_offset_y >= 0 && _frame_offset_y < frame_width)
                    sub_tiled_receptive_field[_field_offset_x][_field_offset_y] = 
                        sub_frame[_frame_offset_x][_frame_offset_y];
            }
        }
    }
}

torch::Tensor cuda_tile_recompute_blocks(
    const torch::Tensor &refer_frame,
    const torch::Tensor &recompute_activation_indices,
    const int &mvs_block_size, const int &offset,
    const int &stride, const int &field_len){
    int dim = refer_frame.dim();
    int frame_height = refer_frame.sizes()[dim-2];
    int frame_width = refer_frame.sizes()[dim-1];
    // (activation_width - 1) * stride + field_len - offset*2 = frame_width
    // int activation_width = (frame_width + offset * 2 - field_len) / stride + 1;

    int channels = refer_frame.sizes()[0];
    auto tiled_pixels = torch::zeros({recompute_activation_indices.sizes()[0], channels,
        field_len, field_len}, refer_frame.options());
    
    // each thread handles (field_len + thread_dim - 1) / thread_dim
    // each block handles a receptive field
    int blocks_x_thread = (field_len + thread_dim_h - 1) / thread_dim_h;
    int blocks_y_thread = (field_len + thread_dim_w - 1) / thread_dim_w;

    dim3 kernel_blocks(channels, recompute_activation_indices.sizes()[0]);
    dim3 threads_per_block(thread_dim_h, thread_dim_w);

    AT_DISPATCH_ALL_TYPES(refer_frame.scalar_type(),
    "cuda_tile_recompute_blocks_kernel", ([&]{
    cuda_tile_recompute_blocks_kernel<<<kernel_blocks, threads_per_block>>>(
        refer_frame.packed_accessor32<scalar_t, 3>(),
        recompute_activation_indices.packed_accessor32<short, 2>(),
        tiled_pixels.packed_accessor32<scalar_t, 4>(),
        mvs_block_size, offset, stride, frame_height, frame_width, blocks_x_thread, blocks_y_thread);
    }));
    return tiled_pixels.contiguous();
}