import time
from typing import List, Dict, Callable, Tuple
import asyncio
from functools import partial

import numpy as np
from functools import partial
from threading import Thread
from queue import Queue
import torch
from .hook_tensor import OffloadProfile, TorchOPProfile, InputSlot
from ._utils import iterate_tensor, log_dur, iterate_all_close
from .test_asyncio import AsyncTCPMessageStream


def fill_slot(arg: torch.Tensor, slots_iter):
    for slot in next(slots_iter):
        slot.container[slot.index] = arg

def fill_slot_cuda(arg: torch.Tensor, slots_iter):
    for slot in next(slots_iter):
        slot.container[slot.index] = arg.cuda()

def fill_output_to_input_slots(tensors, output_dict: Dict[int, List[InputSlot]]):
    slots_iter = iter(output_dict.values())
    def fill_slot(arg: torch.Tensor):
        for slot in next(slots_iter):
            slot.fill(arg)
    iterate_tensor(tensors, fill_slot)

def empty_input_slots(input_slots: List[InputSlot]):
    for slot in input_slots:
        slot.empty()

def slice_output(tensors, send_slice: List[slice], keep_slice: List[slice]):
    send_tensor = []    # Always send a list of tensor
    send_slice_iter = iter(send_slice)
    keep_slice_iter = iter(keep_slice)
    def slice_tensor(arg: torch.Tensor):
        send_tensor.append(arg[next(send_slice_iter)])
        return arg[next(keep_slice_iter)].contiguous()
    keep_tensor = iterate_tensor(tensors, slice_tensor)
    return keep_tensor, send_tensor

def cat_output(tensors, recv_tensor: List[torch.Tensor], keep_slice: List[slice], order: int=0, dim=-1):
    recv_tensor_iter = iter(recv_tensor)
    keep_slice_iter = iter(keep_slice)
    if order == 0:
        def cat_tensor(arg: torch.Tensor):
            return torch.cat([arg[next(keep_slice_iter)], next(recv_tensor_iter)], dim)
    else:
        def cat_tensor(arg: torch.Tensor):
            return torch.cat([next(recv_tensor_iter), arg[next(keep_slice_iter)]], dim)
    return iterate_tensor(tensors, cat_tensor)

def align_tensor_shapes(obj, local_dim, align_mode=1):
    tensors = []
    iterate_tensor(obj, tensors.append)
    shape_len = len(tensors[0].shape)
    align_dim_len = min(t.shape[local_dim] for t in tensors)
    if align_mode == 1:
        slices = [slice(None)] * local_dim + [slice(align_dim_len)] + [slice(None)] * (shape_len - local_dim - 1)
    else:   # 2 at the server side
        slices = [slice(None)] * local_dim + [slice(-align_dim_len, None)] + [slice(None)] * (shape_len - local_dim - 1)
    def _align_shapes(t: torch.Tensor):
        return t[slices]
    return iterate_tensor(obj, _align_shapes)

def compile_plan_to_static_exec(
    profile_result: OffloadProfile, plans: Dict[int, Dict[str, list]],
    sock: AsyncTCPMessageStream, log=print, sleep_step=10, merge=True):
    async def plain_skip(p: TorchOPProfile):
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def skip_with_recv(p: TorchOPProfile):
        with log_dur(sock.add_suffix_to_log, prefix=f"op {p.idx} {p.func_name} recv"):
            recved_tensor = await sock.queued_recv()
        iterate_tensor(recved_tensor,
                    partial(fill_slot_cuda,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def plain_exec(p: TorchOPProfile, align_shape):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def merged_plain_exec(
        ps: List[TorchOPProfile], align_shapes: List[int], sleep_plan: List[bool]):
        with log_dur(sock.add_suffix_to_log, prefix=f"op {ps[0].idx}-{ps[-1].idx} merged exec"):
            for p, align_shape, sleep in zip(ps, align_shapes, sleep_plan):
                if align_shape:
                    args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shapes)
                else:
                    args, kwargs = p.func_args
                intermediates: torch.Tensor = p.func(*args, **kwargs)
                iterate_tensor(intermediates,
                    partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
                for slot in p.input_slots:
                    slot.container[slot.index] = None
                if sleep:
                    await asyncio.sleep(0.)
            torch.cuda.synchronize()
    async def exec_with_recv(p: TorchOPProfile, keep_slice, cat_order, cat_dim, align_shape: int):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(sock.add_suffix_to_log, prefix=f"op {p.idx} {p.func_name} recv"):
            recv_intermediates = await sock.queued_recv()
            intermediates = cat_output(
                intermediates, recv_intermediates,
                keep_slice=keep_slice,
                order=cat_order, dim=cat_dim)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    async def exec_with_offload(p: TorchOPProfile, send_slice, keep_slice, align_shape: List[int]):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(sock.add_suffix_to_log, prefix=f"op {p.idx} {p.func_name} send"):
            intermediates, send_intermediates = slice_output(
                intermediates, send_slice=send_slice, keep_slice=keep_slice)
            await sock.queued_send(send_intermediates)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def _plan_to_static_exec(skip: list, offload: list, recv: list,
                send_slice: dict, send_keep_slice: dict, recv_keep_slice: dict,
                cat_dim: dict, cat_order: dict, recv_first: dict,
                align_shape: list, **kwargs):
        func_calls = []
        sleep_count = 0
        for p in profile_result.profile.values():
            idx = p.idx
            if skip[idx] and np.all(skip[p.input_from]) and not recv[idx]:
                continue    # Ignore completely skipped op

            if skip[idx]:
                if recv[idx]:
                    func_calls.append([skip_with_recv, [p]])
                else:
                    func_calls.append([plain_skip, [p]])
            elif offload[idx]:
                func_calls.append([
                    exec_with_offload, [p, send_slice[idx], send_keep_slice[idx], align_shape[idx]]])
            elif recv[idx]:
                func_calls.append([
                    exec_with_recv, [p, recv_keep_slice[idx], cat_order[idx], cat_dim[idx], align_shape[idx]]])
            else:
                func_calls.append([plain_exec, [p, align_shape[idx]]])
        merged_op = []
        merged_align_shape = []
        ret_func_calls = []
        in_plain_exec = False
        for func_call, args in func_calls:
            if func_call is plain_exec and merge:
                in_plain_exec = True
                merged_op.append(args[0])
                merged_align_shape.append(args[1])
            else:
                if in_plain_exec:
                    if len(merged_op) > 3:
                        sleep_plan = np.zeros(len(merged_op), dtype=bool)
                        sleep_plan[::sleep_step] = True
                        sleep_count += sum(sleep_plan)
                        ret_func_calls.append([merged_plain_exec, [merged_op, merged_align_shape, sleep_plan.tolist()]])
                    else:
                        for op, _align_shape in zip(merged_op, merged_align_shape):
                            ret_func_calls.append([plain_exec, [op, _align_shape]])
                    merged_op = []
                    merged_align_shape = []
                ret_func_calls.append([func_call, args])
                in_plain_exec = False
        if len(merged_op) > 0:
            if len(merged_op) > 3:
                sleep_plan = np.zeros(len(merged_op), dtype=bool)
                sleep_plan[::sleep_step] = True
                sleep_count += sum(sleep_plan)
                ret_func_calls.append([merged_plain_exec, [merged_op, merged_align_shape, sleep_plan.tolist()]])
            else:
                for op, _align_shape in zip(merged_op, merged_align_shape):
                    ret_func_calls.append([plain_exec, [op, _align_shape]])
        return ret_func_calls, sleep_count
    for bw, plan in plans.items():
        profile_result.exec_plan[bw], sleep_count = _plan_to_static_exec(**plan)
        to_offload = np.nonzero(plan["offload"])
        to_recv = np.nonzero(plan["recv"])
        est_time = plan["est_time"]
        log(f"bw {bw}MB/s offload at {to_offload[0].tolist()} recv at {to_recv[0].tolist()} sleep for {sleep_count} ops est time {est_time:.4f}s.")


@torch.no_grad()
async def random_exec_compiled(profile_result: OffloadProfile, bw: int):
    # list(map(lambda x: x[0](*x[1]), profile_result.exec_plan[bw]))
    for exec_func, args in profile_result.exec_plan[bw]:
        await exec_func(*args)
    torch.cuda.synchronize()
    return profile_result.ret_store


@torch.no_grad()
async def random_exec(profile_result: OffloadProfile,
                sock: AsyncTCPMessageStream, log: Callable[[str], None],
                skip: list, offload: list, recv: list,
                send_slice: dict, send_keep_slice: dict, recv_keep_slice: dict,
                cat_dim: dict, cat_order: dict, recv_first: dict,
                align_shape: list, **kwargs):
    """Execute each operator involved in the profile result sequentially.

    Args:
        profile_result (OffloadProfile): profile result of the forward of a model
        sock (AsyncTCPMessageStream)
        log (callable): function to log str
        skip (list): skip plan
        offload (list): offload plan
        recv (list): recv plan
        send_slice (dict): send_slice plan
        send_keep_slice (dict): send_keep_slice plan
        recv_keep_slice (dict): recv_keep_slice plan
        cat_dim (dict): cat_dim plan
        cat_order (dict): cat_order plan

    Returns:
        _type_: _description_
    """
    # Start exec
    for p, _skip, _offload, _recv, _align_shape in zip(
            profile_result.profile.values(), skip, offload, recv, align_shape):
        idx = p.idx
        if _skip:    # Server should always skip the input (first) op
            if _recv:
                with log_dur(log, prefix=f"op {idx} {p.func_name} recv"):
                    recved_tensor = await sock.queued_recv()
                fill_output_to_input_slots(recved_tensor, p.output_idx_slots)
            empty_input_slots(p.input_slots)
        else:     # Client should never skip the input (first) op
            args, kwargs = p.func_args
            if _align_shape:
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
        # await asyncio.sleep(0.)
        if idx % 10 == 0:
            await asyncio.sleep(0.)
    return profile_result.ret_store


def local_random_exec_profile(profile_result: OffloadProfile, log=print):
    torch.cuda.synchronize()
    last_stamp = time.time()
    stime = last_stamp
    # Start exec
    for p in profile_result.profile.values():
        if p.func:
            args, kwargs = p.func_args
            intermediates: torch.Tensor = p.func(*args, **kwargs)
        else:
            intermediates = p.func_args

        fill_output_to_input_slots(intermediates, p.output_idx_slots)
        empty_input_slots(p.input_slots)
        torch.cuda.synchronize()
        c_stamp = time.time()
        p.ops_time = c_stamp - last_stamp
        last_stamp = c_stamp
    log(f"total time from exec step by step with torch.cuda.synchronize: {time.time() - stime:.4e}s.")
    return profile_result.ret_store

async def local_random_exec(profile_result: OffloadProfile):
    # Start exec
    for i, p in enumerate(profile_result.profile.values()):
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        # fill arguments for following ops
        fill_output_to_input_slots(intermediates, p.output_idx_slots)
        empty_input_slots(p.input_slots)
    torch.cuda.synchronize()
    return profile_result.ret_store
