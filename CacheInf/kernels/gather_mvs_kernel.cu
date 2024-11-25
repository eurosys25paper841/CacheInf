#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


const int cuda_block_dim = 16;
const float max_mse_factor = 1.1;


__global__ void cuda_gather_mvs_kernel(
	const torch::PackedTensorAccessor32<short,3> mvs,
	const torch::PackedTensorAccessor32<float,2> mse,
	torch::PackedTensorAccessor32<float,3> activation_mvs,
	torch::PackedTensorAccessor32<float,2> activation_mse,
	torch::PackedTensorAccessor32<float,2> activation_mvs_variance,
	torch::PackedTensorAccessor32<bool,2> activation_recompute_mask,
    const int mvs_block_size, const int offset, const int stride, const int field_edge_blocks,
    const int activation_height, const int activation_width,
    const int blocks_x, const int blocks_y, const float psnr_threshold,
    const float x_factor,  const float y_factor, // scaling factors
    const float data_range
){
    int a_x = threadIdx.x + blockIdx.x * blockDim.x;    // x idx on activation
    int a_y = threadIdx.y + blockIdx.y * blockDim.y;    // y idx on activation

    if (a_x < activation_height && a_y < activation_width){
        int bottom_x, bottom_y;   // x, y idx of corner on input img
        bottom_x = a_x * stride - offset;   // offset is positive
        bottom_y = a_y * stride - offset;

        float acc_mv_x = 0.;
        float mean_mv_x = 0.;
        float acc_mv_y = 0.;
        float mean_mv_y = 0.;
        float acc_mse_mv_x = 0.;
        float acc_mse_mv_y = 0.;
        float acc_mse = 0.;
        float max_mse = 0.;
        float _mse = 0.;
        short _mvs_x, _mvs_y;

        int block_x, block_y, blocks_in_field=0;
        for (int i=0; i < field_edge_blocks; i++)
            for (int j=0; j < field_edge_blocks; j++){
                // // weight each mvs block by intersection area
                block_x = (bottom_x + i * mvs_block_size) / mvs_block_size;
                block_y = (bottom_y + j * mvs_block_size) / mvs_block_size;
                if (block_x >= 0 && block_y >= 0 && block_x < blocks_x && block_y < blocks_y){

                    // We don't care about intersection area since mse is averaged over the block and the block is small enough;
                    _mse = mse[block_x][block_y] + 0.000001;
                    _mvs_x = mvs[block_x][block_y][0];
                    _mvs_y = mvs[block_x][block_y][1];

                    acc_mse += _mse;
                    acc_mv_x += _mvs_x;
                    acc_mv_y += _mvs_y;
                    acc_mse_mv_x += _mse * _mvs_x;
                    acc_mse_mv_y += _mse * _mvs_y;
                    max_mse = max(_mse, max_mse);
                    blocks_in_field++;
                }
            }
        assert(acc_mse > 0);
        mean_mv_x = acc_mv_x / (blocks_in_field + 0.000001);
        mean_mv_y = acc_mv_y / (blocks_in_field + 0.000001);
        
        float variance = 0.;
        for (int i=0; i < field_edge_blocks; i++)
            for (int j=0; j < field_edge_blocks; j++){
                // // weight each mvs block by intersection area
                block_x = (bottom_x + i * mvs_block_size) / mvs_block_size;
                block_y = (bottom_y + j * mvs_block_size) / mvs_block_size;
                if (block_x >= 0 && block_y >= 0 && block_x < blocks_x && block_y < blocks_y){
                    _mse = mse[block_x][block_y] + 0.000001;
                    _mvs_x = mvs[block_x][block_y][0];
                    _mvs_y = mvs[block_x][block_y][1];
                    variance += ((_mvs_x - mean_mv_x) * (_mvs_x - mean_mv_x) + (_mvs_y - mean_mv_y) * (_mvs_y - mean_mv_y)) * (max_mse - _mse);
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x==0 && threadIdx.y == 0){
                    //     printf("variance %.4f [%d,%d] mvs_x %d (%.4f) mvs_y %d (%.4f) a_x %d a_y %d\n",
                    //      variance, block_x, block_y, _mvs_x, mean_mv_x, _mvs_y, mean_mv_y, a_x, a_y);
                    // }
                }
            }
        activation_mvs_variance[a_x][a_y] = variance / (blocks_in_field + 0.000001);
        float avg_mv_x, avg_mv_y, avg_mse;
        // acc_mv_x *= (max_mse * max_mse_factor);    // acc_mv_x = \sum_i max_mse * \sigma * mv_i_x
        // acc_mv_y *= (max_mse * max_mse_factor);
        // avg_mv_x = (acc_mv_x - acc_mse_mv_x) / (max_mse * max_mse_factor * blocks_in_field - acc_mse);
        // // avg_mv_x = acc_mse_mv_x / acc_mse;
        // avg_mv_y = (acc_mv_y - acc_mse_mv_y) / (max_mse * max_mse_factor * blocks_in_field - acc_mse);
        // // avg_mv_y = acc_mse_mv_y / acc_mse;
        // // avg_mv_x = \fract{\sum_i (max_mse * \sigma - mse_i) * mv_i_x }{\sum_i (max_mse * \sigma - mse_i)}

        activation_mvs[a_x][a_y][0] = acc_mv_x / (blocks_in_field + 0.000001) * x_factor; 
        activation_mvs[a_x][a_y][1] = acc_mv_y / (blocks_in_field + 0.000001) * y_factor;

        avg_mse = acc_mse / (blocks_in_field + 0.000001);
        activation_mse[a_x][a_y] = avg_mse;
        // https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/metrics/simple_metrics.py#L112-L168 psnr calculation
        float psnr = 10 * log10(data_range * data_range / avg_mse);
        if (psnr < psnr_threshold) // larger mse leads to smaller psnr
            activation_recompute_mask[a_x][a_y] = true;
    }
}

__global__ void cuda_reflect_mvs_kernel(
	torch::PackedTensorAccessor32<bool,2> mvs_recompute_mask,
	const torch::PackedTensorAccessor32<short,3> mvs,
	const torch::PackedTensorAccessor32<float,2> mse,
	const torch::PackedTensorAccessor32<bool,2> activation_recompute_mask,
    const int mvs_block_size, const int offset, const int stride, const int field_edge_blocks,
    const int activation_height, const int activation_width,
    const int blocks_x, const int blocks_y, const float psnr_threshold,
    const float data_range){
    // From activation mvs and activation mse, infer the actual executed mvs of input image blocks
    // Each thread is a mvs block on the input image
    int b_x = threadIdx.x + blockIdx.x * blockDim.x;    // x idx on mvs blocks
    int b_y = threadIdx.y + blockIdx.y * blockDim.y;    // y idx on mvs blocks
    if (b_x < blocks_x && b_y < blocks_y){
        // int b_x = (a_x_start * stride - offset) / mvs_block_size + field_edge_blocks - 1
        // stride is smaller than mvs_block_size
        int a_x_start = max(((b_x - field_edge_blocks + 1) * mvs_block_size + (mvs_block_size -1) + offset) / stride, 0);
        int a_y_start = max(((b_y - field_edge_blocks + 1) * mvs_block_size + (mvs_block_size -1) + offset) / stride, 0);
        if (a_x_start < activation_height && a_y_start < activation_width)
        {
            int a_x_end = min((b_x * mvs_block_size + mvs_block_size -1 + offset) / stride + 1, activation_height);
            int a_y_end = min((b_y * mvs_block_size + mvs_block_size -1 + offset) / stride + 1, activation_width);
            float self_mse =  mse[b_x][b_y];
            float self_psnr = 10 * log10(data_range * data_range / self_mse);
            bool needs_recompute = false;
            for (int i = a_x_start; i < a_x_end; i++)
                for (int j = a_y_start; j < a_y_end; j++){
                    if (activation_recompute_mask[i][j])
                        needs_recompute = true;
                }
            if (needs_recompute)
                mvs_recompute_mask[b_x][b_y] = true;
            }
    }
}


std::vector<torch::Tensor> cuda_gather_mvs(
    const int &activation_height, const int &activation_width,
    const torch::Tensor &mvs, const torch::Tensor &mse,
    const int &mvs_block_size, const int &offset, const int &stride, const int &field_len,
    const float &psnr_threshold, const float &data_range, const int &topk_edge){
    // mvs: [blocks_x, blocks_y,  2], dtype short; mse: [blocks_x, blocks_y], dtype float
    const int mvs_blocks_x = mvs.sizes()[0], mvs_blocks_y = mvs.sizes()[1];
    
    float activation_x_scale = float(activation_height) / float(mvs_block_size * mvs_blocks_x);
    float activation_y_scale = float(activation_width) / float(mvs_block_size * mvs_blocks_y);

    const auto bool_options = torch::TensorOptions()
		.dtype(torch::kBool)
		.layout(torch::kStrided)
		.device(torch::kCUDA, 0)
		.requires_grad(false);
    
    // const auto short_options = torch::TensorOptions()
	// 	.dtype(torch::kShort)
	// 	.layout(torch::kStrided)
	// 	.device(torch::kCUDA, 0)
	// 	.requires_grad(false);

    const auto float_options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.layout(torch::kStrided)
		.device(torch::kCUDA, 0)
		.requires_grad(false);
    // activation_mvs will produce fractional motion vectors
    auto activation_mvs = torch::zeros({activation_height, activation_width, 2}, float_options);
    auto activation_mse = torch::zeros({activation_height, activation_width}, float_options);
    auto activation_mvs_variance = torch::zeros({activation_height, activation_width}, float_options);
    auto activation_recompute_mask = torch::zeros({activation_height, activation_width}, bool_options);

    // Each cuda block handles cuda_block_dim**2 activation pixels
    // Each cuda thread handles a receptive filed of an activation pixel
    const int cuda_blocks_x = (activation_height + cuda_block_dim - 1) / cuda_block_dim;
    const int cuda_blocks_y = (activation_width + cuda_block_dim - 1) / cuda_block_dim;
    const dim3 Kernel_Blocks(cuda_blocks_x, cuda_blocks_y);
    const dim3 Threads_Per_Block(cuda_block_dim, cuda_block_dim);
    // int combined_receptive_field_length = (stride - 1) * cuda_block_dim + field_len;
    // int blocks_in_combined_receptive_field = std::pow((combined_receptive_field_length + mvs_block_size - 1) / mvs_block_size, 2);
    // std::cout<< mvs.sizes() << std::endl;
    // std::cout<< mse.sizes() << std::endl;

    cuda_gather_mvs_kernel<<<Kernel_Blocks, Threads_Per_Block>>>(
        mvs.packed_accessor32<short, 3>(), mse.packed_accessor32<float, 2>(),
        activation_mvs.packed_accessor32<float, 3>(), activation_mse.packed_accessor32<float, 2>(),
        activation_mvs_variance.packed_accessor32<float, 2>(),
        activation_recompute_mask.packed_accessor32<bool, 2>(),
        mvs_block_size, offset, stride, field_len / mvs_block_size, activation_height, activation_width,
        mvs_blocks_x, mvs_blocks_y, psnr_threshold, activation_x_scale, activation_y_scale,
        data_range
    );

    // // if (activation_recompute_mask.sum().item().toInt() > topk_edge*topk_edge){
    //     auto col_topk = activation_mvs_variance.topk(topk_edge, 0);
    //     auto col_topk_val = std::get<0>(col_topk);
    //     auto col_topk_indices = std::get<1>(col_topk);
    //     auto _col_topk_indices = torch::stack({
    //         col_topk_indices,
    //         torch::arange(
    //             col_topk_indices.sizes()[1], col_topk_indices.options()).repeat({topk_edge, 1})
    //         },
    //         -1);
    //     auto row_topk = col_topk_val.topk(topk_edge, 1);
    //     auto row_topk_val = std::get<0>(row_topk);
    //     auto row_topk_indices = std::get<1>(row_topk);
    //     auto _row_topk_indices = torch::stack({
    //         torch::arange(
    //             topk_edge, row_topk_indices.options()).repeat({topk_edge, 1}).transpose(0,1),
    //         row_topk_indices
    //         }, -1);    // [topk_edge, topk_edge, 2]
    //     // std::cout << _col_topk_indices.sizes() << std::endl;
    //     // std::cout << _row_topk_indices.sizes() << std::endl;
    //     std::vector<torch::indexing::TensorIndex> indices;
    //     for (int64_t i = 0; i < _row_topk_indices.size(2); ++i) {
    //         indices.push_back(_row_topk_indices.index({"...", i}));
    //     }
    //     auto _activation_recompute_indices = _col_topk_indices.index(indices);  // [topk_edge, topk_edge, 2]
        
    //     std::vector<torch::indexing::TensorIndex> indices2;
    //     for (int64_t i = 0; i < _activation_recompute_indices.size(2); ++i) {
    //         indices2.push_back(_activation_recompute_indices.index({"...", i}));
    //     }
    //     activation_recompute_mask.zero_();
    //     // std::cout << _activation_recompute_indices.sizes() << std::endl;
    //     activation_recompute_mask.index_put_(indices2, true);
    // auto activation_recompute_indices = torch::nonzero(
    //     activation_recompute_mask).to(torch::kShort, true);


    // Cast the gathered mvs average against mse back to the input image blocks to determine the whether an input image block needs to be recomputed
    // mse of activation over the psnr_threshold will cause the corresponding area of mvs block to be recomputed
    // auto mvs_recompute_mask = torch::zeros({mvs_blocks_x , mvs_blocks_y}, bool_options);
    // const dim3 Reflect_Blocks((mvs_blocks_x + cuda_block_dim -1) / cuda_block_dim,
    //     (mvs_blocks_y + cuda_block_dim -1) / cuda_block_dim);
    // cuda_reflect_mvs_kernel<<<Reflect_Blocks, Threads_Per_Block>>>(
    //     mvs_recompute_mask.packed_accessor32<bool, 2>(),
    //     mvs.packed_accessor32<short, 3>(), mse.packed_accessor32<float, 2>(),
    //     activation_recompute_mask.packed_accessor32<bool, 2>(),
    //     mvs_block_size, offset, stride, field_len / mvs_block_size,
    //     activation_height, activation_width,
    //     mvs_blocks_x, mvs_blocks_y, psnr_threshold, data_range
    // );

    // auto block_recompute_indices = torch::nonzero(mvs_recompute_mask).to(torch::kShort, true);
    // return {activation_mvs.contiguous(), activation_mse.contiguous(), activation_mvs_variance.contiguous(),
    //     activation_recompute_indices.contiguous(), block_recompute_indices.contiguous()};
    return {activation_mvs.contiguous(), activation_mse.contiguous(), activation_mvs_variance.contiguous()};
}