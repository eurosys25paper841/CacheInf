#include <vector>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input tensor must be a contiguous cuda tensor")

torch::Tensor decode_mvs(
    const torch::Tensor &refer_frame, const torch::Tensor &mvs, const int &block_size, const bool &bilinear);

std::vector<torch::Tensor> extract_mvs(
    const torch::Tensor &ref_frame, const torch::Tensor &query_frame, const int &block_size);

torch::Tensor fill_activations(torch::Tensor _activations, torch::Tensor _activation_updates,
    const torch::Tensor &activation_update_indices);

std::vector<torch::Tensor> gather_mvs(
    const int &activation_h, const int &activation_w,
    const torch::Tensor &mvs, const torch::Tensor &mse,
    const int &mvs_block_size, const int &offset, const int &stride, const int &field_len,
    const float &psnr_threshold, const float &data_range, const int &topk_edge);

torch::Tensor tile_recompute_blocks(
    const torch::Tensor &refer_frame,
    const torch::Tensor &recompute_activation_indices,
    const int &mvs_block_size, const int &offset, const int &stride, const int &field_len);

torch::Tensor merge_update_pixels(
    torch::Tensor refer_frame, const torch::Tensor &update_pixels,
    const torch:: Tensor &update_block_indices, const int &block_size);

torch::Tensor get_pixels_by_blocks(
    const torch::Tensor &refer_frame, 
    const torch:: Tensor &block_indices, const int &block_size);

using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Module to construct corelation between an input image/activation with its descendent activation through spatial computations such as convolution. It then detects motions in the input image/activation to reuse the descendent activation to skip the computation.";

    m.def("decode_mvs", &decode_mvs, "put blocks in refer frame into update_frame to keep track of the unchanged activations based on motion vectors; for recomputed activations, the update frame is unchanged. The function also returns the sparse pixels corresponding to the recomputed activations.",
    "refer_frame"_a, "mvs"_a, "block_size"_a, "bilinear"_a);


    m.def("extract_mvs", &extract_mvs, "extract motion vectors with three step block matching algorithm",
    "ref_frame"_a, "query_frame"_a, "block_size"_a);

    m.def("fill_activations", &fill_activations, "Update activations with partial activation updates which is stacked along the batch size dimension according to its indices.",
    "activations"_a, "activation_updates"_a, "update_indices"_a);

    m.def("gather_mvs", &gather_mvs, "gather motion vectors for an activation tensor",
    "activation_height"_a, "activation_width"_a, "mvs"_a, "mse"_a, "mvs_block_size"_a, "offset"_a, "stride"_a, "field_len"_a, "psnr_threshold"_a, "data_range"_a, "topk_edge"_a);

    m.def("tile_recompute_blocks", &tile_recompute_blocks, "From recompute pixels reconstruct receptive files and stack them along the batch size dimension to leverage the acceleration of underlying cuda kernels.",
    "refer_frame"_a, "recompute_activation_indices"_a,
    "mvs_block_size"_a, "offset"_a, "stride"_a, "field_len"_a);

    m.def("merge_update_pixels", &merge_update_pixels, "Inplace merging update_pixels indexed by update_block_indices into refer_frame", "refer_frame"_a, "update_pixels"_a, "update_block_indices"_a, "block_size"_a);

    m.def("get_pixels_by_blocks", &get_pixels_by_blocks, "Extract pixels indexed by block_indices from refer frame", "refer_frame"_a, "block_indices"_a, "block_size"_a);
}