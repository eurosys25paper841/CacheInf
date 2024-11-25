#include <vector>
#include <torch/extension.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input tensor must be a contiguous cuda tensor")
// CUDA forward declaration
torch::Tensor cuda_tile_recompute_blocks(
    const torch::Tensor &refer_frame,
    const torch::Tensor &recompute_activation_indices,
    const int &mvs_block_size, const int &offset,
    const int &stride, const int &field_len);



/// @return A vector of two tensors: 1. merged frame by a. using pixels from refer_frame moved by mvs and b. pixels from recompute_pixels indexed by recompute_block_indices (since they are to be recomputed); 2. tiled receptive fields for recomputation
torch::Tensor tile_recompute_blocks(
    const torch::Tensor &_refer_frame,
    const torch::Tensor &recompute_activation_indices,
    const int &mvs_block_size, const int &offset, const int &stride, const int &field_len) {
    // [C,H,W] or [C,D,H,W]
    auto refer_frame = _refer_frame.squeeze(0);
    CHECK_INPUT(refer_frame);
    CHECK_INPUT(recompute_activation_indices);

    if (refer_frame.dim() == 4){  // [C,D,H,W]
        int C = refer_frame.sizes()[0];
        int D = refer_frame.sizes()[1];
        auto _refer_frame = refer_frame.flatten(0, 1);

        // return [N, C*D, field_len, field_len] -> [N, C, D, field_len, field_len]
        return cuda_tile_recompute_blocks(
            _refer_frame, recompute_activation_indices,
            mvs_block_size, offset, stride, field_len).unflatten(1, {C,D});
    }
    else
      return cuda_tile_recompute_blocks(
        refer_frame, recompute_activation_indices,
        mvs_block_size, offset, stride, field_len); // [N, C, field_len, field_len]
}


