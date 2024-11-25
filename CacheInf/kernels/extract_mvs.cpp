#include <vector>
#include <torch/extension.h>

// CUDA forward declaration
std::vector<torch::Tensor> cuda_extract_mvs(
    const torch::Tensor &ref_frame, const torch::Tensor &query_frame, const int &block_size);

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input tensor must be a contiguous cuda tensor")

std::vector<torch::Tensor> extract_mvs(
    const torch::Tensor &_ref_frame, const torch::Tensor &_query_frame, const int &block_size) {
    auto ref_frame = _ref_frame.squeeze(0);
    auto query_frame = _query_frame.squeeze(0);
    CHECK_INPUT(ref_frame);
    CHECK_INPUT(query_frame);
	int dims = ref_frame.dim();
	TORCH_CHECK((dims <= 4) && (dims >= 3)); // [C,H,W] or [C,D,H,W], only batch size == 1 supported
    TORCH_CHECK(ref_frame.sizes() == query_frame.sizes());
    TORCH_CHECK(ref_frame.dtype() == query_frame.dtype());
    TORCH_CHECK((ref_frame.sizes()[dims-1] % block_size == 0) &&
        (ref_frame.sizes()[dims-2] % block_size == 0));
	if (dims == 4){
		TORCH_CHECK(ref_frame.sizes()[0] == 1);
        auto _ref_frame = ref_frame.index({0});
        auto _query_frame = query_frame.index({0});
        return cuda_extract_mvs(_ref_frame, _query_frame, block_size);
	}
    return cuda_extract_mvs(ref_frame, query_frame, block_size);
}

