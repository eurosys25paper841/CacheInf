import time
from typing import List, Dict, Callable, Tuple
import asyncio
from functools import partial
import matplotlib.pyplot as plt
import cv2
import numpy as np
from functools import partial
from queue import Queue
from multiprocessing import Queue as mpQueue
import torch
from .hook_tensor import OffloadProfile, TorchOPProfile, InputSlot
from ._utils import iterate_tensor, log_dur, iterate_all_close
from .test_asyncio import AsyncTCPMessageStream
from .random_exec import fill_slot, slice_output, cat_output, empty_input_slots, fill_output_to_input_slots
from .cached_schedule import MAX_CACHE_X, CACHE_FACTOR

empty_preamble = b"0"*int(1024*1024*0.5) # 0.5MB
# Utilities to detect motions of pixel blocks, transfer the motions to cached_activations of a certain layer,
# use the motions to manipulate the input images and the cached_activations, gather recompute input and merge recompute cached_activations back to the cached_activations
from cacheinf_cuda import (extract_mvs, gather_mvs, decode_mvs, fill_activations,
                           get_pixels_by_blocks, merge_update_pixels, tile_recompute_blocks)
from .cached_schedule import ROBOT, SERVER
from .look_ahead import Kalman2DTracker


class CacheInfo:
    mvs_block_size: int=4
    block_num_to_x: float = None
    cached_input_image_height: int = None
    cached_input_image_width: int = None
    cached_input_image: torch.Tensor = None
    cache_op: int = None
    min_stride: int = 64
    fixed_cache_op: int = None
    cached_activations_height: int = None
    cached_activations_width: int = None
    cached_activations: torch.Tensor = None
    local_dim: int = None
    # activation_x * stride - offset = input_image_x
    # activation_x_end * stride - offset + field_len = input_image_x_end + offset
    offset: int = None
    stride: int = None
    field_len: int = None
    feature_slice: slice = None
    flatten_orig_shape = None

    cache_location: int = None
    psnr_threshold = 10.
    # The data range of the input image (distance between minimum and maximum possible values).
    data_range = 1.
    top_k: int = 10
    key: tuple = None

async def ranged_random_exec(profile_result: OffloadProfile, sock: AsyncTCPMessageStream,
                      start: int, end: int, cache_info: CacheInfo, 
                      log_info="with cache", activation_indices: torch.Tensor=None):
    ps = profile_result.profile
    intermediates = None
    with log_dur(sock.add_suffix_to_log, prefix=f"op {start}-{end} {log_info} exec"):
        intermediates = None
        for i in range(start, end):
            p = ps[i]
            args, kwargs = p.func_args
            # TODO
            if False and p.align_shape and cache_info.cache_op != 0 and activation_indices is not None:
                args = list(args)
                arg_2: torch.Tensor = args[1]
                local_dim = p.local_dim
                size_h = args[0].shape[local_dim-1]
                size_w = args[0].shape[local_dim]
                axis_h = torch.arange(0, size_h, 1).to(arg_2.device)
                axis_w = torch.arange(0, size_w, 1).to(arg_2.device)
                grid = torch.meshgrid(axis_h, axis_w, indexing='ij')    # [size, size]
                grid = torch.stack([grid[0], grid[1]], dim=-1) # [size, size, 2]
                height_scale = arg_2.shape[local_dim - 1]/cache_info.cached_activations_height 
                width_scale = arg_2.shape[local_dim]/cache_info.cached_activations_width
                activation_indices[:, 0] = (height_scale * activation_indices[:, 0]).type(torch.short)
                activation_indices[:, 1] = (width_scale * activation_indices[:, 1]).type(torch.short)
                grid_indices = activation_indices.transpose(0,1).unsqueeze(-1).unsqueeze(-1) + torch.moveaxis(grid,2,0).unsqueeze(1)  # [2, N, size, size]
                arg_2 = torch.moveaxis(arg_2, (local_dim-1, local_dim), (0,1))
                aligned_arg_2 = arg_2[tuple(grid_indices)].squeeze(3)   # exclude the original batch dim
                aligned_arg_2 = torch.moveaxis(aligned_arg_2, (1,2), (local_dim-1, local_dim))
                assert aligned_arg_2.shape[local_dim-1] == size_h
                assert aligned_arg_2.shape[local_dim] == size_w
                args[1] = aligned_arg_2
            try:
                intermediates: torch.Tensor = p.func(*args, **kwargs)
            except Exception as e:
                raise e
            iterate_tensor(intermediates,
                partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
            for slot in p.input_slots:
                slot.container[slot.index] = None
            if i % 10 == 0:
                await asyncio.sleep(0.)
        # torch.cuda.synchronize()    # TODO
        return intermediates

async def HP_random_exec(profile_result: OffloadProfile,
                sock: AsyncTCPMessageStream, log: Callable[[str], None],
                skip: list, offload: list, recv: list,
                send_slice: dict, send_keep_slice: dict, recv_keep_slice: dict,
                cat_dim: dict, cat_order: dict, recv_first: dict,
                align_shape: list, start: int, end: int, cache_info="no cache", **kwargs):
    """Execute each operator involved in the profile result sequentially.
    """
    # Start exec
    intermediates = None
    with log_dur(sock.add_suffix_to_log, prefix=f"op {start}-{end} {cache_info} exec"):
        profiles = profile_result.profile
        for idx in range(start, end):
            p = profiles[idx]
            idx = p.idx
            _skip, _offload, _recv, _align_shape = skip[idx], offload[idx], recv[idx], align_shape[idx]
            if _skip:    # Server should always skip the input (first) op
                if _recv:
                    with log_dur(log, prefix=f"op {idx} {p.func_name} recv"):
                        intermediates = await sock.queued_recv()
                    fill_output_to_input_slots(intermediates, p.output_idx_slots)
                empty_input_slots(p.input_slots)
            else:     # Client should never skip the input (first) op
                args, kwargs = p.func_args
                if _align_shape and False:  # TODO
                    local_dim = p.local_dim
                    arg0, arg1 = args
                    align_dim_len = min(arg0.shape[local_dim], arg1.shape[local_dim])
                    if _align_shape == 1:
                        slices = [slice(None)] * local_dim + [slice(align_dim_len)] + [slice(None)] * (len(arg0.shape) - local_dim - 1)
                    else:   # 2 at the server side
                        slices = [slice(None)] * local_dim + [slice(-align_dim_len, None)] + [slice(None)] * (len(arg0.shape) - local_dim - 1)
                    if arg0.shape[local_dim] < arg1.shape[local_dim]:
                        intermediates: torch.Tensor = p.func(arg0, arg1[slices], **kwargs)
                    else:
                        intermediates: torch.Tensor = p.func(arg0[slices], arg1, **kwargs)
                else:
                    try:
                        intermediates: torch.Tensor = p.func(*args, **kwargs)
                    except Exception as e:
                        print(p)

                if _recv or _offload:
                    if _recv and recv_first[idx]:
                        with log_dur(log, prefix=f"op {idx} {p.func_name} recv"):
                            recv_intermediates = await sock.queued_recv()
                        # merge received tensor with intermediates
                        intermediates = cat_output(
                            intermediates, recv_intermediates,
                            keep_slice=recv_keep_slice[idx],
                            order=cat_order[idx], dim=cat_dim[idx])
                    if _offload:
                        intermediates, send_intermediates = slice_output(
                            intermediates,
                            send_slice=send_slice[idx],
                            keep_slice=send_keep_slice[idx])
                        with log_dur(log, prefix=f"op {idx} {p.func_name} send"):
                            await sock.queued_send(send_intermediates)
                    if _recv and not recv_first[idx]:
                        with log_dur(
                            log, prefix=f"op {idx} {p.func_name} recv; keep slice {recv_keep_slice[idx]} cat order {cat_order[idx]}"):
                            recv_intermediates = await sock.queued_recv()
                        # merge received tensor with intermediates
                        intermediates = cat_output(
                            intermediates, recv_intermediates,
                            keep_slice=recv_keep_slice[idx],
                            order=cat_order[idx], dim=cat_dim[idx])
                # fill arguments for following ops
                iterate_tensor(intermediates,
                        partial(fill_slot,
                                slots_iter=iter(p.output_idx_slots.values())))
                # fill_output_to_input_slots(intermediates, p.output_idx_slots)
                for slot in p.input_slots:
                    slot.container[slot.index] = None
                # empty_input_slots(p.input_slots)
            if idx % 10 == 0:
                await asyncio.sleep(0.)
        return intermediates

import numpy as np
import numba

@numba.jit(nopython=True)
def _cluster_top_mvs_variance(var: np.ndarray, start_x: int, start_y: int):
    height = var.shape[-2]
    width = var.shape[-1]
    end_x = start_x + 1
    end_y = start_y + 1
    _start_x = start_x
    _start_y = start_y
    _end_x = start_x + 1
    _end_y = start_y + 1
    while True:
        for i in range(-1, 2):
            for j in range(-1, 2):
                for curr_x in range(start_x, end_x):
                    _curr_x = curr_x + i
                    for _curr_y in [start_y, end_y-1]:
                        if _curr_x >= 0 and _curr_y >= 0 and \
                            _curr_x < height and _curr_y < width and var[_curr_x][_curr_y] > 0.:
                            _start_x = min(start_x, _curr_x, _start_x)
                            _start_y = min(start_y, _curr_y, _start_y)
                            _end_x = max(end_x, _curr_x + 1, _end_x)
                            _end_y = max(end_y, _curr_y + 1, _end_y)
                            var[_curr_x][_curr_y] = 0.
                for curr_y in range(start_y, end_y):
                    _curr_y = curr_y + j
                    for _curr_x in [start_x, end_x-1]:
                        if _curr_x >= 0 and _curr_y >= 0 and \
                            _curr_x < height and _curr_y < width and var[_curr_x][_curr_y] > 0.:
                            _start_x = min(start_x, _curr_x, _start_x)
                            _start_y = min(start_y, _curr_y, _start_y)
                            _end_x = max(end_x, _curr_x + 1, _end_x)
                            _end_y = max(end_y, _curr_y + 1, _end_y)
                            var[_curr_x][_curr_y] = 0.
        if not (_start_x == start_x and _start_y == start_y and _end_x == end_x and _end_y == end_y):
            start_x = min(start_x, _start_x)
            start_y = min(start_y, _start_y)
            end_x = max(end_x, _end_x)
            end_y = max(end_y, _end_y)
        else:
            break
    return [start_x, end_x, start_y, end_y]


empirical_h_pad = 2
def cluster_top_activation_mvs_variance(var: torch.Tensor,
                                        offset: int, stride: int, field_len: int,
                                        img_height: int, img_width: int, min_stride=128):
    var: np.ndarray = var.cpu().numpy()
    height = var.shape[-2]
    width = var.shape[-1]
    bar = np.max(var) * 0.5
    var[var < bar] = 0.
    # median = np.median(var)
    # var[var < median] = 0.  # Ignore smaller than median
    idx = var.argmax()
    idx_x, idx_y = idx // var.shape[-1], idx % var.shape[-1]
    start_x, end_x, start_y, end_y = _cluster_top_mvs_variance(var, idx_x, idx_y)
    # Find a minimum block to contain these activations
    _stride_h = img_height // height
    _stride_w = img_width // width
    x_num = end_x - start_x
    y_num = end_y - start_y
    pad_x = min_stride // _stride_h * (x_num % _stride_h > 1) -  x_num % (min_stride  // _stride_h) + empirical_h_pad * min_stride // _stride_h
    pad_y = min_stride // _stride_w * (y_num % _stride_w > 1) -  y_num % (min_stride  // _stride_w)
    pad_x = min(pad_x, height - x_num)
    pad_y = min(pad_y, width - y_num)
    if start_x <= pad_x // 2:
        x_lower = start_x
    elif height - end_x <= pad_x // 2:
        x_lower = pad_x - (height - end_x)
    else:
        x_lower = min(start_x, pad_x//2)
    x_upper = pad_x - x_lower
    if start_y <= pad_y // 2:
        y_lower = start_y
    elif width - end_y <= pad_y // 2:
        y_lower = pad_y - (width - end_y)
    else:
        y_lower = min(start_y, pad_y//2)
    y_upper = pad_y - y_lower
    new_x_start = start_x - x_lower
    new_x_end = end_x + x_upper
    new_y_start = start_y - y_lower
    new_y_end = end_y + y_upper
    field_height = int((new_x_end - new_x_start)/height * img_height)
    field_width = int((new_y_end - new_y_start) / width * img_width)
    assert new_x_start >= 0 and new_y_start >= 0
    assert new_x_end <= height and new_y_end <= width
    assert field_height % min_stride == 0
    assert field_height % min_stride == 0
    coord_start_x = int(new_x_start / height * img_height)
    coord_end_x = int(new_x_end / height * img_height)
    coord_start_y = int(new_y_start / width * img_width)
    coord_end_y = int(new_y_end / width * img_width)
    field_slice_height = field_slice_width = slice(None)
    input_slice_height = slice(int(new_x_start / height * img_height),
                               int(new_x_end/height * img_height))
    input_slice_width = slice(int(new_y_start / width * img_width),
                               int(new_y_end/ width * img_width))
    return torch.tensor([[new_x_start, new_x_end, new_y_start, new_y_end]]), [field_slice_height, field_slice_width], [input_slice_height, input_slice_width], [field_height, field_width]


debug = True
async def robot_cached_inference(
    idx: int, profile_result: OffloadProfile,
    sock: AsyncTCPMessageStream, cache_info: CacheInfo, schedule: dict, offload_schedule: dict,
    order_queue: mpQueue, obs_queue: mpQueue, predict_queue: mpQueue, update: bool, local):
    # At the robot side: extract mvs of input image and gather mvs for activation;
    #                       find cached_activations and pixel blocks needed to recompute
    tensors = []
    iterate_tensor(profile_result.profile[0].func_args, tensors.append)
    assert len(tensors) == 1
    new_input_image: torch.tensor = tensors[0]

    last_cached_image = cache_info.cached_input_image
    mvs_block_size = cache_info.mvs_block_size
    cached_activations = cache_info.cached_activations
    psnr_threshold, data_range = cache_info.psnr_threshold, cache_info.data_range
    a_h, a_w = cache_info.cached_activations_height, cache_info.cached_activations_width
    offset, stride, field_len = cache_info.offset, cache_info.stride, cache_info.field_len
    cached_op = cache_info.cache_op
    fixed_cache_op = cache_info.fixed_cache_op
    local_dim = cache_info.local_dim
    last_location = cache_info.cache_location
    feature_slice = cache_info.feature_slice
    
    _current_bw = sock.last_bandwidth
    current_bw = min(int(_current_bw), sock.max_bw)

    recompute = False
    whole_send = False
    compute_ratio = 1.
    if last_cached_image is not None:
        mvs, mse = extract_mvs(last_cached_image, new_input_image, mvs_block_size)
        x = (((10* torch.log10(data_range*data_range / mse)) < psnr_threshold).sum() / mse.numel()).item()
        # topk_edge = int(np.sqrt(a_h * a_w * x)/4)
        topk_edge = 2

        (activations_mvs, activations_mse, activation_mvs_variance) = \
            gather_mvs(
                a_h, a_w,
                mvs, mse,
                mvs_block_size, offset, stride, field_len,
                cache_info.psnr_threshold, cache_info.data_range, topk_edge)
        sock.log("Entering Cluster")
        (activations_recompute_indices, field_slices, input_image_slices,
         field_input_areas) = cluster_top_activation_mvs_variance(
            activation_mvs_variance, offset, stride, field_len, last_cached_image.shape[-2], last_cached_image.shape[-1], cache_info.min_stride)
        compute_ratio = np.prod(field_input_areas) / np.prod(last_cached_image.shape[-2:])

        sock.log("Entering Cluster")
        sock.add_suffix_to_log(
            f"Gathered recompute bbox {input_image_slices} ({field_input_areas}) compute_ratio {compute_ratio:.4f}\n")
        topk_mse = torch.topk(activations_mse.ravel(), int(activations_mse.numel()*0.1))[0][-1]
        recompute_block_indices = torch.nonzero(
            (mse>topk_mse) &
            (10*torch.log10(data_range**2/mse) < psnr_threshold))
        recompute_block_pixels = get_pixels_by_blocks(new_input_image, recompute_block_indices, mvs_block_size)
        if not update and (10*torch.log10(data_range**2/activations_mse)).mean() < psnr_threshold:
            recompute = True
            whole_send = True
    else:
        x = 1.
        recompute = True
        whole_send = True
    if not update:
        if (10* torch.log10(data_range*data_range / mse)).mean() < psnr_threshold:
            recompute = True
            whole_send = True
    key = tuple([current_bw, min(int(x*CACHE_FACTOR), MAX_CACHE_X-1), cached_op, last_location])
    new_cached_op, new_location = schedule[key]
    new_cached_op = fixed_cache_op
    if new_location == ROBOT:
        sock.skip_next_recv()
    sock.recording_log.prefix = f"{idx} th with cache sock: "
    if local:
        sock.recording_log.send_num = 0
        sock.recording_log.recv_num = 0
    elif whole_send:
        sock.recording_log.send_num = offload_schedule[key[0]]["send_num"] + 1
        sock.recording_log.recv_num = max(offload_schedule[key[0]]["recv_num"], 1)
    else:
        sock.recording_log.send_num = 2
        sock.recording_log.recv_num = 1

    sock.add_suffix_to_log(f"Observed bw {_current_bw:.2f}MB/s x {x:.4f} schedule cached op {new_cached_op} new location {new_location}")

    # new_cached_op == 0 -> should not use cache -> recompute & wholly send
    if new_cached_op == 0 or new_cached_op != cached_op or new_location != last_location or \
        x * CACHE_FACTOR > 70:
        cache_info.cache_location = new_location
        cache_info.cache_op = new_cached_op
        recompute = True
        whole_send = True
    await sock.queued_send([key, recompute, whole_send])
        # if new_cached_op != 0 and new_cached_op != cached_op:  # TODO fix cache_info
        #     _p = profile_result.profile[new_cached_op]
        #     p = profile_result.profile[_p.input_from[0]]
        #     local_dim = p.local_dim
        #     cache_info.cache_op = new_cached_op
        #     cache_info.offset, cache_info.stride, cache_info.field_len = p.offset, p.stride, p.field_len
        #     cache_info.cached_activations_height, cache_info.cached_activations_width = \
        #         p.input_shapes[0][local_dim-1:local_dim+1]
        #     cache_info.local_dim = local_dim

    # TODO select recompute pixel blocks
    if not whole_send:
        selected_recompute_block_indices = recompute_block_indices # TODO
        recompute_block_pixels = get_pixels_by_blocks(
            new_input_image, selected_recompute_block_indices, mvs_block_size)
        if not update:
            selected_recompute_block_indices = selected_recompute_block_indices[:0]
            recompute_block_pixels = recompute_block_pixels[:0]

    if new_location == SERVER:  # offload
        if whole_send:     # Need to wholly transmit
            current_offload_schedule = offload_schedule[current_bw]
            await sock.queued_send(new_input_image)
            ret = await sock.queued_recv()
            iterate_tensor(ret,
                        partial(fill_slot,
                                slots_iter=iter(
                                    profile_result.profile[profile_result.end_idx].output_idx_slots.values())))
            # await HP_random_exec(profile_result, sock, sock.add_suffix_to_log, start=0, end=profile_result.end_idx+1, **current_offload_schedule)

            local_ratio = current_offload_schedule["x"][1]
            cache_info.cached_input_image = new_input_image
            # if local_ratio > 0.01:
            #     pad_size = int(np.around(
            #         cache_info.cached_input_image_width * local_ratio))
            #     cache_info.cached_input_image[..., :pad_size] = -100.
        else:
            # TODO Only sends mvs, activations_recompute_indices
            # selected_recompute_block_indices, recompute_block_pixels
            await sock.queued_send([mvs, activations_mvs,
                                    activations_recompute_indices, field_slices, input_image_slices, field_input_areas,
                                    selected_recompute_block_indices, recompute_block_pixels])
            if update:
                moved_cached_frame = decode_mvs(last_cached_image,
                                                mvs, mvs_block_size, False)
                cache_info.cached_input_image = merge_update_pixels(
                    moved_cached_frame, recompute_block_pixels,
                    selected_recompute_block_indices, mvs_block_size).contiguous()
                # if debug:
                #     cv2.imwrite(f"{idx}_moved.jpg", moved_cached_frame[0].moveaxis(0, -1).cpu().numpy()[..., ::-1])
                #     cv2.imwrite(f"{idx}_merged.jpg", cache_info.cached_input_image[0].moveaxis(0, -1).cpu().numpy()[..., ::-1])

            recved_tensor = await sock.queued_recv()
            iterate_tensor(recved_tensor,
                    partial(fill_slot,
                            slots_iter=iter(profile_result.profile[profile_result.end_idx].output_idx_slots.values())))
    else:
        # replace inference input with tiled receptive fields for subsequent computation
        if not recompute:
            _new_activations = decode_mvs(cached_activations, activations_mvs, 1, True)
            if update:
                compute_ratio = np.prod(field_input_areas) / np.prod(last_cached_image.shape[-2:])
                moved_cached_frame = decode_mvs(last_cached_image,
                                                mvs, mvs_block_size, False)
                new_cached_frame = merge_update_pixels(
                    moved_cached_frame, recompute_block_pixels,
                    selected_recompute_block_indices, mvs_block_size).contiguous()
                cache_info.cached_input_image = new_cached_frame

                _input = torch.zeros(list(moved_cached_frame.shape[0:-2]) + field_input_areas, dtype=moved_cached_frame.dtype, device=moved_cached_frame.device)
                _input[..., field_slices[0], field_slices[1]] = new_cached_frame[
                    ..., input_image_slices[0], input_image_slices[1]]  # [N,C,H,W]

                profile_result.profile[0].input_slots[0].fill(_input)

                if _input.numel() > 0:
                    await ranged_random_exec(
                        profile_result, sock, 0, new_cached_op, cache_info,
                        f"with cache prefix compute ratio {compute_ratio:.4f}", activations_recompute_indices)
                    _activations_update: torch.Tensor = profile_result.profile[new_cached_op].input_slots[0].tensor

                    activations_update = torch.moveaxis(
                        _activations_update, (local_dim-1,local_dim), (-2,-1)).flatten(1, -3)
                    _new_activations = fill_activations(
                        _new_activations,
                        activations_update, activations_recompute_indices)
                cache_info.cached_activations = _new_activations.contiguous()

            new_activations = torch.moveaxis(
                _new_activations.unflatten(1, cache_info.flatten_orig_shape),
                (-2,-1), (local_dim-1,local_dim)).contiguous()

            profile_result.profile[new_cached_op].input_slots[0].fill(new_activations)
        else:
            cache_info.cached_input_image = new_input_image
            await ranged_random_exec(
                profile_result, sock, 0, fixed_cache_op, cache_info,
                "no cache (change cache idx) compute ratio 1.")
            if update:
                cache_info.cached_activations = torch.moveaxis(
                    profile_result.profile[fixed_cache_op].input_slots[0].tensor,
                    (local_dim-1,local_dim), (-2,-1)).flatten(1, -3).contiguous()

        await ranged_random_exec(profile_result, sock,
            fixed_cache_op, profile_result.end_idx+1, cache_info, "no cache suffix")


async def server_cached_inference(
    idx: int, profile_result: OffloadProfile,
    sock: AsyncTCPMessageStream, cache_info: CacheInfo,
    schedule: dict, offload_schedule: dict, update: bool, local):
    # At the server side: receive mvs of input image and gather mvs for activation;
    #                       find cached_activations and pixel blocks needed to recompute
    last_cached_image = cache_info.cached_input_image
    mvs_block_size = cache_info.mvs_block_size
    cached_activations = cache_info.cached_activations
    a_h, a_w = cache_info.cached_activations_height, cache_info.cached_activations_width
    offset, stride, field_len = cache_info.offset, cache_info.stride, cache_info.field_len
    cached_op = cache_info.cache_op
    fixed_cache_op = cache_info.fixed_cache_op
    local_dim = cache_info.local_dim
    last_location = cache_info.cache_location

    key, recompute, whole_send = await sock.queued_recv()
    current_bw = key[0]
    x = key[1] / CACHE_FACTOR

    new_cached_op, new_location = schedule[key]
    if new_location == ROBOT:
        if sock.sending:    # Last sock send msg unfinished
            await sock.queued_send(b"")
        else:
            await sock.queued_send(empty_preamble)
    else:
        cache_info.cache_op = new_cached_op
        cache_info.cache_location = new_location
    if local:
        sock.recording_log.send_num = 0
        sock.recording_log.recv_num = 1
    elif new_cached_op == 0:
        sock.recording_log.prefix = f"{idx} th sock no cache: "
        sock.recording_log.send_num = max(offload_schedule[current_bw]["send_num"], 1)
        sock.recording_log.recv_num = offload_schedule[current_bw]["recv_num"] + 1
    else:
        sock.recording_log.prefix = f"{idx} th sock with cache: "
        sock.recording_log.send_num = 1
        sock.recording_log.recv_num = 2
    current_offload_schedule = offload_schedule[current_bw]
    cache_info.cache_op, cache_info.cache_location = new_cached_op, new_location

    sock.add_suffix_to_log(f"Observed bw {current_bw:.2f}MB/s x {x:.4f} schedule cached op {new_cached_op} new location {new_location}")

    if new_location == SERVER:  # offload
        # new_cached_op == 0 -> should not use cache -> recompute & wholly send
        if whole_send:
            recv = await sock.queued_recv()
            profile_result.profile[0].input_slots[0].fill(recv)
            partial_new_input_image = profile_result.profile[0].func_args[0][0]
            # partial_new_input_image = (await HP_random_exec(
            #     profile_result, sock, sock.add_suffix_to_log, start=0, end=1, **current_offload_schedule))[0]

            # server pad [... 0:x*height] with zero
            recved_ratio = current_offload_schedule["x"][0]
            if recved_ratio < 0.99 and False:
                pad_size = int(np.around(
                    cache_info.cached_input_image_width * (1- recved_ratio)))
                cache_info.cached_input_image = torch.nn.functional.pad(
                    partial_new_input_image, [0, 0, pad_size, 0], "constant", 0.)
                assert cache_info.cached_input_image.shape[-1] == cache_info.cached_input_image_width
            else:
                cache_info.cached_input_image = partial_new_input_image
            await ranged_random_exec(
                profile_result, sock, 0, fixed_cache_op,
                cache_info=f"no cache compute ratio 1.0 ")

            # await HP_random_exec(profile_result, sock, sock.add_suffix_to_log, start=1,
            #                end=fixed_cache_op, cache_info=f"no cache compute ratio 1.0 ", **current_offload_schedule)
            partial_activations = profile_result.profile[fixed_cache_op].input_slots[0].tensor
            activation_ratio = current_offload_schedule["x"][fixed_cache_op]

            cache_info.cached_activations = torch.moveaxis(partial_activations, (local_dim-1,local_dim), (-2,-1)).flatten(1, -3).contiguous()

            ret = await ranged_random_exec(
                profile_result, sock, fixed_cache_op, profile_result.end_idx+1,
                cache_info=f"no cache compute ratio 1.0 ")
            await sock.queued_send(ret)
            
            # await HP_random_exec(profile_result, sock, sock.add_suffix_to_log, start=fixed_cache_op,
            #                end=profile_result.end_idx+1, **offload_schedule[current_bw])

        else:
            # Only recv mvs, activations_recompute_indices
            # selected_recompute_block_indices, recompute_block_pixels
            (mvs, activations_mvs, activations_recompute_indices, field_slices, input_image_slices, field_input_areas,
             selected_recompute_block_indices, recompute_block_pixels) = await sock.queued_recv()

            if not recompute:
                _new_activations = decode_mvs(cached_activations, activations_mvs, 1, True)
                if update:
                    compute_ratio = np.prod(field_input_areas) / np.prod(last_cached_image.shape[-2:]) 
                    moved_cached_frame = decode_mvs(last_cached_image,
                                                    mvs, mvs_block_size, False)
                    new_cached_frame = merge_update_pixels(
                        moved_cached_frame, recompute_block_pixels,
                        selected_recompute_block_indices, mvs_block_size).contiguous()
                    cache_info.cached_input_image = new_cached_frame
                    
                    _input = torch.zeros(list(moved_cached_frame.shape[:-2]) + field_input_areas, dtype=moved_cached_frame.dtype, device=moved_cached_frame.device)
                    _input[..., field_slices[0], field_slices[1]] = new_cached_frame[
                        ..., input_image_slices[0], input_image_slices[1]]  # [N,C,H,W]

                    if _input.numel() > 0:
                        profile_result.profile[0].func_args[0][0] = _input

                        await ranged_random_exec(
                            profile_result, sock, 0, fixed_cache_op, cache_info,
                            f"with cache prefix compute ratio {compute_ratio:.4f}", activations_recompute_indices)
                        _activations_update: torch.Tensor = profile_result.profile[
                            fixed_cache_op].input_slots[0].tensor

                        activations_update = torch.moveaxis(
                            _activations_update, (local_dim-1,local_dim), (-2,-1)).flatten(
                                1, -3)

                        _new_activations = fill_activations(
                            decode_mvs(cached_activations, activations_mvs, 1, True),
                            activations_update, activations_recompute_indices)
                    cache_info.cached_activations = _new_activations.contiguous()
                new_activations = torch.moveaxis(_new_activations.unflatten(
                    1, cache_info.flatten_orig_shape), (-2,-1), (local_dim-1,local_dim)).contiguous()

                profile_result.profile[fixed_cache_op].input_slots[0].fill(new_activations)
            else:
                profile_result.profile[0].func_args[0][0] = new_cached_frame
                await ranged_random_exec(
                    profile_result, sock, 0, fixed_cache_op, cache_info,
                    "no cache (change cache idx) compute ratio 1.")

            await ranged_random_exec(profile_result, sock,
                fixed_cache_op, profile_result.end_idx+1, cache_info, "no cache suffix")
            await sock.queued_send(profile_result.ret_store)
    else:
        # replace inference input with tiled receptive fields for subsequent computation
        cache_info.cached_input_image = None
        cache_info.cached_activations = None
