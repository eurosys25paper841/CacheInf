import torch

import cacheinf_cuda
import time

# width + offset * 2 = stride * (activation_width - 1) + field_len
refer_frame = torch.randint(0, 255, [1, 3,16,16], device="cuda:0") / 255
refer_activation = torch.randn(1, 13, 4, 4, device="cuda:0")
mvs_block_size, offset, stride, field_len = 4, 4, 4, 12
h = refer_frame.shape[-2]
w = refer_frame.shape[-1]
query_frame = torch.randint(0, 255, [1, 3,16,16], device="cuda:0") / 255
query_frame[..., h//2 - h//4:, w//2 - w // 4:] = refer_frame[..., :h//2 + h//4, :w//2 + h//4]
stime = time.time()
# Extract motion vectors by comparing refer_frame and query_frame
mvs, mse = cacheinf_cuda.extract_mvs(refer_frame, query_frame, mvs_block_size)
# print(f"mvs ({mvs.shape}): {mvs}")
# print(f"mse {mse.shape}: {mse}")

(activation_mvs, activation_mse, activation_recompute_indices, recompute_block_indices) = \
    cacheinf_cuda.gather_mvs(refer_activation, mvs, mse, mvs_block_size,
                             offset, stride, field_len, 10., 1.)    # top 100
# print(f"activation_mvs ({activation_mvs.shape}): {activation_mvs}")
# print(f"activation_mse ({activation_mse.shape}): {activation_mse}")
# print(f"activation_recompute_indices ({activation_recompute_indices.shape}): {activation_recompute_indices}")
# print(f"recompute_block_indices ({recompute_block_indices.shape}): {recompute_block_indices}")

# move refer frame by motion vectors
moved_refer_frame = cacheinf_cuda.decode_mvs(refer_frame, mvs, mvs_block_size)
# print(f"moved_refer_frame ({moved_refer_frame.shape}): {moved_refer_frame}")

assert torch.allclose(moved_refer_frame[..., h//2 - h//4:, w//2 - w // 4:], refer_frame[..., :h//2 + h//4, :w//2 + h//4], atol=1e-6)

# Extract pixels that needs to recompute
recompute_pixels = cacheinf_cuda.get_pixels_by_blocks(query_frame, recompute_block_indices, mvs_block_size)

# print(f"query_frame ({query_frame.shape}): {query_frame}")
# print(f"recompute_pixels ({recompute_pixels.shape}): {recompute_pixels}")

# # Merge recompute pixels to moved reference frame to represent the next frame after moving activations and recompute
# # merged_update_frame acts as the reference frame of the next inference
merged_update_frame = cacheinf_cuda.merge_update_pixels(moved_refer_frame, recompute_pixels,
                                                        recompute_block_indices, mvs_block_size)
assert torch.allclose(merged_update_frame, query_frame, atol=1e-6)
print(f"merged_update_frame ({merged_update_frame.shape}): {merged_update_frame}")

moved_activations = cacheinf_cuda.decode_mvs(refer_activation, activation_mvs, 1)
print(f"moved_activations ({moved_activations.shape}): {moved_activations}")

tiled_input_to_recompute = cacheinf_cuda.tile_recompute_blocks(
    merged_update_frame.contiguous(), activation_recompute_indices.contiguous(), mvs_block_size, offset, stride, field_len)
print(f"tiled_input_to_recompute ({tiled_input_to_recompute.shape}): {tiled_input_to_recompute}")


torch.cuda.synchronize()
print(f"dur {time.time() - stime:.4f}s")

for _ in range(100):
    stime = time.time()
    mvs, mse = cacheinf_cuda.extract_mvs(refer_frame, query_frame, mvs_block_size)

    (activation_mvs, activation_mse, activation_recompute_indices, recompute_block_indices) = \
        cacheinf_cuda.gather_mvs(refer_activation, mvs, mse, mvs_block_size,
                                offset, stride, field_len, 10., 1.)    # top 100

    # move refer frame by motion vectors
    moved_refer_frame = cacheinf_cuda.decode_mvs(refer_frame, mvs, mvs_block_size)

    # Extract pixels that needs to recompute
    recompute_pixels = cacheinf_cuda.get_pixels_by_blocks(query_frame, recompute_block_indices, mvs_block_size)


    # # Merge recompute pixels to moved reference frame to represent the next frame after moving activations and recompute
    # # merged_update_frame acts as the reference frame of the next inference
    merged_update_frame = cacheinf_cuda.merge_update_pixels(moved_refer_frame, recompute_pixels,
                                                            recompute_block_indices, mvs_block_size)

    moved_activations = cacheinf_cuda.decode_mvs(refer_activation, activation_mvs, 1)

    tiled_input_to_recompute = cacheinf_cuda.tile_recompute_blocks(
        merged_update_frame.contiguous(), activation_recompute_indices.contiguous(), mvs_block_size, offset, stride, field_len)

    torch.cuda.synchronize()
    print(f"dur {time.time() - stime:.4f}s")


"""
(0,0) (0,1) (0,2) (0,3)
(1,0) (1,1) (1,2) (1,3)
(2,0) (2,1) (2,2) (2,3)
(3,0) (3,1) (3,2) (3,3)



 (0,2) (0,3)
 (1,2) (1,3)
(2,0) (2,1) (0,0) (0,1)
(3,0) (3,1) (1,0) (1,1)
"""
