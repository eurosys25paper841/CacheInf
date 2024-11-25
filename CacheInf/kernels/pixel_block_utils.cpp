#include <vector>
#include <torch/extension.h>


torch::Tensor gen_mask_block_indices(
    torch::Tensor refer_frame, const torch:: Tensor &update_block_indices,
    const int &block_size){
    // [C,H,W]
    // const int channels = refer_frame.sizes()[0];
    const int frame_height = refer_frame.sizes()[1];
    const int frame_width = refer_frame.sizes()[2];
    const int blocks_y = frame_width / block_size;
    const auto bool_options = torch::TensorOptions()
		.dtype(torch::kBool)
		.layout(torch::kStrided)
		.device(torch::kCUDA, 0)
		.requires_grad(false);
    auto update_block_mask = torch::zeros(
        {frame_height/block_size, 1, frame_width/block_size, 1}, bool_options); // [blocks_x, 1, blocks_y, 1]
    auto indices_1d = update_block_indices.index({"...", 0}) * blocks_y + update_block_indices.index({"...", 1});
    update_block_mask.view(-1).index_put_({indices_1d.to(torch::kLong, true)}, true);

    return update_block_mask.expand({-1, block_size, -1, block_size}).reshape({frame_height, frame_width});
}


torch::Tensor _merge_blocks(
    torch::Tensor refer_frame, const torch::Tensor &update_pixels,
    const torch:: Tensor &update_block_indices, const int &block_size){

    auto update_pixel_mask = gen_mask_block_indices(refer_frame, update_block_indices, block_size);

    return refer_frame.moveaxis(0, -1).index_put({update_pixel_mask}, update_pixels).moveaxis(-1, 0);
    // return refer_frame;
}

/// @brief Inplace merging update_pixels indexed by update_block_indices into refer_frame
torch::Tensor merge_update_pixels(
    torch::Tensor _refer_frame, const torch::Tensor &update_pixels,
    const torch:: Tensor &update_block_indices, const int &block_size){
    if (update_block_indices.sizes()[0]==0)
        return _refer_frame;
    auto refer_frame = _refer_frame.squeeze(0);
    if (refer_frame.dim() == 4){    // [C,D,H,W]
        int C = refer_frame.sizes()[0];
        int D = refer_frame.sizes()[1];
        auto _refer_frame = refer_frame.flatten(0, 1);
        return _merge_blocks(_refer_frame, update_pixels, update_block_indices, block_size).unflatten(0, {C,D}).unsqueeze(0).contiguous();
    }
    return _merge_blocks(refer_frame, update_pixels, update_block_indices, block_size).unsqueeze(0).contiguous();
}

torch::Tensor _avg_blocks(
    torch::Tensor &refer_frame, const torch::Tensor &update_pixels,
    const torch:: Tensor &update_block_indices, const int &block_size){
    // update_block_indices: [M, 2]
    // refer_frame: [C, H, W]
    std::vector<int64_t> dim = {1};
    int _dim = refer_frame.dim();
    // int height = refer_frame.size(_dim-2), width = refer_frame.size(_dim-1);
    auto update_dim = update_pixels.dim();
    auto update_height = update_pixels.sizes()[update_dim-2];
    auto update_width = update_pixels.sizes()[update_dim-1];
    int len = update_pixels.size(0);
    int x_start, y_start, x_end, y_end, update_start_x, update_start_y;
    auto ret = refer_frame.clone();
    // int height_start, height_end, width_start, width_end;
    for (int i=0; i < len; i++){
        x_start = std::max(update_block_indices[i][0].item().toInt(), 0);
        x_end = std::min(update_block_indices[i][1].item().toInt(), int(refer_frame.size(1)));
        y_start = std::max(update_block_indices[i][2].item().toInt(), 0);
        y_end = std::min(update_block_indices[i][3].item().toInt(), int(refer_frame.size(2)));
        update_start_x = (update_height - (x_end - x_start)) / 2;
        update_start_y = (update_width - (y_end - y_start)) / 2;
        ret.index_put_({"...", torch::indexing::Slice(x_start, x_end, 1),
            torch::indexing::Slice(y_start, y_end, 1)},
                update_pixels[i].index({"...", torch::indexing::Slice(0, x_end - x_start, 1),
                        torch::indexing::Slice(0, y_end - y_start, 1)}));
    }

    // auto _update_pixels = update_pixels.index({"...", 0, 0});    // [M,C]
    // auto update_pixel_mask = gen_mask_block_indices(refer_frame, update_block_indices, 1); // [H, W]
    // auto update_norm = torch::linalg::vector_norm(_update_pixels, torch::Scalar(2), 1, true, c10::nullopt); // [M, 1]
    // std::slice Slice;

    // auto previous = refer_frame.moveaxis(0, -1).index({update_pixel_mask}); // [M, C]
    // auto previous_norm = torch::linalg::vector_norm(previous, torch::Scalar(2), 1, true, c10::nullopt); // [M, 1]
    // // auto ret = refer_frame.moveaxis(0, -1).index_put({update_pixel_mask},
    // //     (_update_pixels / update_norm * previous_norm + previous)/2.).moveaxis(-1, 0);
    // auto ret = refer_frame.moveaxis(0, -1).index_put({update_pixel_mask},
    //     _update_pixels).moveaxis(-1, 0);
    return ret;
}

/// @brief Inplace averaging update_pixels indexed by update_block_indices into refer_frame
torch::Tensor avg_update_pixels(
    torch::Tensor _refer_frame, const torch::Tensor &update_pixels,
    const torch:: Tensor &update_block_indices, const int &block_size){
    if (update_block_indices.sizes()[0]==0)
        return _refer_frame;
    auto refer_frame = _refer_frame.squeeze(0);
    if (refer_frame.dim() == 4){    // [C,D,H,W]
        int C = refer_frame.sizes()[0];
        int D = refer_frame.sizes()[1];
        auto _refer_frame = refer_frame.flatten(0, 1);
        return _avg_blocks(_refer_frame, update_pixels, update_block_indices, block_size).unflatten(0, {C,D});
    }
    return _avg_blocks(refer_frame, update_pixels, update_block_indices, block_size);
}


torch::Tensor _get_pixels_by_blocks(
    const torch::Tensor &refer_frame, 
    const torch:: Tensor &block_indices, const int &block_size){
    // int channels = refer_frame.sizes()[0];

    auto pixel_mask = gen_mask_block_indices(refer_frame, block_indices, block_size);
    return refer_frame.moveaxis(0, -1).index({pixel_mask});
}

torch::Tensor get_pixels_by_blocks(
    const torch::Tensor &_refer_frame, 
    const torch:: Tensor &block_indices, const int &block_size){
    auto refer_frame = _refer_frame.squeeze(0);
    if (block_indices.sizes()[0]==0){
        if (refer_frame.dim() == 4)
            return torch::zeros({refer_frame.sizes()[0], refer_frame.sizes()[1], 0}, refer_frame.options());
        else
            return torch::zeros({refer_frame.sizes()[0],  0}, refer_frame.options());
    }
    if (refer_frame.dim() == 4){    // [C,D,H,W]
        int C = refer_frame.sizes()[0];
        int D = refer_frame.sizes()[1];
        auto _refer_frame = refer_frame.flatten(0, 1);
        return _get_pixels_by_blocks(_refer_frame, block_indices, block_size).unflatten(0, {C,D});
    }
    return _get_pixels_by_blocks(refer_frame, block_indices, block_size);
}
