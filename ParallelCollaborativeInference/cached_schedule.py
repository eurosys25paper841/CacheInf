import time
from typing import List, Dict, Callable, Tuple
from .hook_tensor import OffloadProfile, TorchOPProfile, InputSlot
import numpy as np

def x_to_cached_comp_time(profile_result: OffloadProfile, cached_op: int, x: float)->float:
    """Estimate the computation time when caching at the cached_op th operator and the input has (1-x) reusable pixels

    Args:
        profile_result (OffloadProfile): _description_
        cached_op (int): _description_
        x (float): _description_

    Returns:
        float: estimated computation time
    """
    num_ops = len(profile_result.profile)
    profiles = profile_result.profile
    area = np.prod(profile_result.profile[0].input_shapes[0][-2:])
    non_reusable_area = area * x
    non_reusable_area_len = np.sqrt(non_reusable_area)
    p = profiles[cached_op]
    if len(profiles[cached_op].input_from) > 0:
        parent_profile = profiles[profiles[cached_op].input_from[0]]
    else:
        parent_profile = p
    local_dim = parent_profile.local_dim
    if local_dim:
        a_h, a_w = parent_profile.output_shapes[0][local_dim-1:local_dim+1]
    else:
        a_h, a_w = parent_profile.output_shapes[0][-2:]
    field_len = parent_profile.field_len
    stride = parent_profile.stride
    offset = parent_profile.offset
    num_activations_len = max((non_reusable_area_len - field_len + stride - 1) // stride, 0)   # TODO
    # how many receptive fields can cover the non_reusable_area
    num_activations = num_activations_len ** 2
    comp_portion = num_activations * field_len * field_len / np.prod(
        profile_result.profile[0].input_shapes[-2:])
    if comp_portion > 0.8:
        return np.inf
    comp_time = comp_portion * np.sum(
        profiles[j].ops_time for j in range(0,cached_op)) + np.sum(
        profiles[j].ops_time for j in range(cached_op,num_ops))
    return comp_time

MAX_CACHE_X=80
CACHE_FACTOR=100.
def optimal_computation_reduction(profile_result: OffloadProfile):
    # Under different input non-reusable ratio, find the optimal layer to cache
    num_ops = len(profile_result.profile)
    profiles = profile_result.profile
    last_local_operator = max([i for i in range(num_ops) if profiles[i].local_dim])
    first_non_local_op = profile_result.profile[last_local_operator].output_to[0]
    # candidates = [i for i in range(num_ops) if (profiles[i].hook_kwargs["conv"] and
    #                                             not profiles[i].masked)] + [first_non_local_op]
    candidates = [0, first_non_local_op]
    optimal_cached_op = {}
    optimal_comp_time = {}
    optimal_cached_op_with_different_pixel_blocks = {}
    local_inference_time = sum(p.ops_time for p in profile_result.profile.values())
    for X in range(0, MAX_CACHE_X):
        optimal_cached_op[X] = 0
        optimal_comp_time[X] = np.inf
        for i in candidates:
            if i == 0:
                comp_time = local_inference_time
            else:
                comp_time = x_to_cached_comp_time(profile_result, i, X/CACHE_FACTOR)
            if comp_time < optimal_comp_time[X]:
                optimal_cached_op[X] = i
                optimal_comp_time[X] = comp_time
    return optimal_cached_op

ROBOT=0
SERVER=1
look_ahead_step=100
def look_ahead_schedule(robot_profile_result: OffloadProfile,
                        server_profile_result: OffloadProfile,
                        offload_schedule, optimal_cached_op, max_bw):
    switch_schedule: Dict[List[int, int, int, int, int, int], List] = {}
    # current_bw, estimated_future_avg_bw, predicted_x cached_op, location
    profiles = robot_profile_result.profile
    num_ops = len(profiles)
    last_local_operator = max(i for i in range(1, num_ops) if profiles[i].local_dim)
    first_non_local_op = robot_profile_result.profile[last_local_operator].output_to[0]
    # candidates = [i for i in range(num_ops) if (profiles[i].hook_kwargs["conv"] and
    #                                             not profiles[i].masked)] + [first_non_local_op]
    candidates = [0, first_non_local_op]
    input_size = robot_profile_result.profile[0].input_size / 1024 / 1024
    output_size = robot_profile_result.profile[robot_profile_result.end_idx].output_size / 1024 / 1024
    local_inference_time = sum(p.ops_time for p in robot_profile_result.profile.values())
    server_inference_time = sum(p.ops_time for p in server_profile_result.profile.values())
    for predicted_bw in range(max_bw+1):
        total_local_time = local_inference_time * look_ahead_step
        total_offload_time = offload_schedule[predicted_bw]['est_time'] * look_ahead_step
        for predicted_X in range(0, MAX_CACHE_X):
            predicted_x = predicted_X / CACHE_FACTOR
            _optimal_cached_op = optimal_cached_op[predicted_X]
            for cached_op in candidates:
                current_robot_time = x_to_cached_comp_time(
                    robot_profile_result, cached_op, predicted_x) *\
                        look_ahead_step
                current_server_time = (x_to_cached_comp_time(
                    server_profile_result, cached_op, predicted_x) +\
                        (input_size * predicted_x /8 + output_size) / (predicted_bw+1e-5)) *\
                            look_ahead_step
                for location in [ROBOT, SERVER]:
                    key = tuple([predicted_bw, predicted_X, cached_op, location])
                    if _optimal_cached_op == 0: # cache seems slower
                        if total_local_time < total_offload_time - 1.0:
                            switch_schedule[key] = [0, ROBOT]
                        else:
                            switch_schedule[key] = [0, SERVER]
                        continue
                    if key not in switch_schedule:
                        switch_schedule[key] = [cached_op, location]

                    # check whether offload and change cached op

                    offload_time = offload_schedule[predicted_bw]['est_time']
                    cached_offload_time = ((
                        input_size * predicted_x + output_size) / (predicted_bw+1e-5) +\
                        x_to_cached_comp_time(
                            server_profile_result,
                            _optimal_cached_op, predicted_x)) *\
                        (look_ahead_step - 1)

                    local_switch_op_time = local_inference_time +\
                        (look_ahead_step - 1) * \
                            x_to_cached_comp_time(
                                robot_profile_result, _optimal_cached_op, predicted_x)
                    server_switch_op_time = server_inference_time +\
                        cached_offload_time + (input_size * predicted_x + output_size) / (predicted_bw+1e-5)
                    if location == ROBOT:
                        if offload_time + cached_offload_time < current_robot_time - 0.2:
                            switch_schedule[key] = [_optimal_cached_op, SERVER]
                        elif local_switch_op_time < current_robot_time - 0.2:
                            switch_schedule[key] = [_optimal_cached_op, ROBOT]
                        else:
                            switch_schedule[key] = [cached_op, ROBOT]
                    else:
                        if local_switch_op_time < current_server_time - 1.0:
                            switch_schedule[key] = [_optimal_cached_op, ROBOT]
                        elif server_switch_op_time < current_server_time - 0.2:
                            switch_schedule[key] = [_optimal_cached_op, SERVER]
                        else:
                            switch_schedule[key] = [cached_op, SERVER]
    return switch_schedule