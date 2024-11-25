#include <vector>
#include <torch/extension.h>

// CUDA forward declaration
std::vector<torch::Tensor> cuda_gather_mvs(
    const int &activation_h, const int &activation_w,
    const torch::Tensor &mvs, const torch::Tensor &mse,
    const int &mvs_block_size, const int &offset, const int &stride, const int &field_len,
    const float &psnr_threshold, const float &data_range, const int &topk_edge);


#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input tensor must be a contiguous cuda tensor")

/// @brief From the motion vectors (mvs) and mean square error (mse) of input image blocks, gather mvs and mse for activations and mark the activations needed to be recomputed. Also reflects the activation mvs to the input image blocks and marks the input image blocks needed for recomputed.
/// @param activation Activation that can be reused
/// @param mvs Motion vectors of input image blocks
/// @param mse Mean square error of input image blocks
/// @param mvs_block_size 
/// @param offset 
/// @param stride Computes as activation_idx_x * stride + offset = receptive_field_corner_x
/// @param field_len Length of an edge of the receptive field
/// @param psnr_threshold Activation psnr that is over psnr_threshold will be marked as needed to be recomputed.
/// @param data_range The data range of the input image (distance between minimum and maximum possible values).
/// @return A vector of five tensors: 1. the gathered activation mvs, 2. the gathered activation mse, 3. indices of activations needed to be recomputed, 5. indices of input image blocks needed to be recomputed.
std::vector<torch::Tensor> gather_mvs(
    const int &activation_h, const int &activation_w,
    const torch::Tensor &mvs, const torch::Tensor &mse,
    const int &mvs_block_size, const int &offset, const int &stride, const int &field_len,
    const float &psnr_threshold, const float &data_range, const int &topk_edge) {
    CHECK_INPUT(mvs);
    CHECK_INPUT(mse);
    TORCH_CHECK(field_len%mvs_block_size == 0)
    TORCH_CHECK(mvs.dim() == 3 && mse.dim() == 2,
        "Expecting mvs.dim() == 3 && mse.dim() == 2, but got mvs.dim() ", mvs.dim(), " and mse.dim() ", mse.dim());
    TORCH_CHECK(offset < field_len,
        "Offset ", offset, " cannot be larger that receptive field length ", field_len);
    TORCH_CHECK((activation_h - 1) * stride + field_len <=
             mvs.sizes()[0] * mvs_block_size + offset * 2 && 
             activation_h * stride + field_len >=
             mvs.sizes()[0] * mvs_block_size + offset * 2 &&
        (activation_w - 1) * stride + field_len <=
            mvs.sizes()[1] * mvs_block_size + offset * 2 &&
        activation_w * stride + field_len >=
            mvs.sizes()[1] * mvs_block_size + offset * 2, 
        "Stride and field len miss-match with the input image size. Activation sizes ",
        activation_h, " ", activation_w, 
        " offset ", offset, " stride ", stride,
        " receptive field length ", field_len, " mvs sizes ", mvs.sizes());
    return cuda_gather_mvs(activation_h, activation_w, mvs, mse, mvs_block_size, offset, stride, field_len, psnr_threshold, data_range, topk_edge);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("gather_mvs", &gather_mvs, "gather motion vectors for an activation tensor");
// }
