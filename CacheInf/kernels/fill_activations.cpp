#include <vector>
#include <torch/extension.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input tensor must be a contiguous cuda tensor")

torch::Tensor merge_update_pixels(
    torch::Tensor refer_frame, const torch::Tensor &update_pixels,
    const torch:: Tensor &update_block_indices, const int &block_size);

torch::Tensor avg_update_pixels(
    torch::Tensor _refer_frame, const torch::Tensor &update_pixels,
    const torch:: Tensor &update_block_indices, const int &block_size);


torch::Tensor fill_activations(torch::Tensor _activations, torch::Tensor activation_updates,
    const torch::Tensor &activation_update_indices){
    // activations [C,H,W] or [C,D,H,W] where batch size is one
    // activation_updates [M,C,1,1] or [M,C,D,1,1]
    auto activations = _activations.squeeze(0);
    int dim = activations.dim();
    // auto update_dim = _activation_updates.dim();
    // int h = _activation_updates.sizes()[update_dim-2], w = _activation_updates.sizes()[update_dim-1];
    // auto activation_updates = _activation_updates.index({"...", {{h/2}}, {{w/2}}});
    // TORCH_CHECK(activation_updates.sizes()[dim-1] == 1 && activation_updates.sizes()[dim] == 1,
    //     "Expecting activation_updates to have shapes in [M,C,1,1] or [M,C,D,1,1], but got ", activation_updates.sizes());
    try
    {if (dim == 4){  // [C,D,H,W]
        int C = activations.sizes()[0];
        int D = activations.sizes()[1];
        auto _activations = activations.flatten(0,1);
        auto _activation_updates = activation_updates.flatten(1,2).transpose(0, 1); // [M,C,...]
        auto ret = avg_update_pixels(_activations, _activation_updates, activation_update_indices, 1);
        return ret.unflatten(0, {C,D}).unsqueeze(0).contiguous();
    }
    else{
        return avg_update_pixels(activations, activation_updates, activation_update_indices, 1).unsqueeze(0).contiguous();
    }}
    catch (...) {
        std::cerr << "activations shape" << activations.sizes() << std::endl;
        std::cerr << "activation_updates shape " << activation_updates.sizes() << std::endl;
        std::cerr << "activation_update_indices " << activation_update_indices << std::endl;
        throw "Fill activation error";
    }
}
