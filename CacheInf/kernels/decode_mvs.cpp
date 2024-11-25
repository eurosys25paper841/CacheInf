#include <vector>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// CUDA forward declaration
torch::Tensor cuda_decode_mvs(
    const torch::Tensor &refer_frame, const torch::Tensor &mvs, const int &block_size, const bool &bilinear);

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input tensor must be a contiguous cuda tensor")

/// @return moved frame by using pixels from refer_frame moved by mvs
torch::Tensor decode_mvs(
    const torch::Tensor &_refer_frame, const torch::Tensor &mvs, const int &block_size, const bool &bilinear) {
    // [C,H,W] or [C,D,H,W]
    CHECK_INPUT(_refer_frame);
    auto dtype = _refer_frame.dtype();
    auto refer_frame = _refer_frame.squeeze(0).to(torch::kFloat32, true);
    if (refer_frame.dim() == 4){  // [C,D,H,W]
        int C = refer_frame.sizes()[0];
        int D = refer_frame.sizes()[1];
        auto _refer_frame = refer_frame.flatten(0, 1);

        return cuda_decode_mvs(_refer_frame, mvs.to(torch::kFloat), block_size, bilinear).unflatten(0, {C,D}).unsqueeze(0).to(dtype, true);
    }
    else
      return cuda_decode_mvs(refer_frame, mvs.to(torch::kFloat), block_size, bilinear).unsqueeze(0).to(dtype, true);
}



