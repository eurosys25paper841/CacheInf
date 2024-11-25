#! /usr/bin/env python
import sys
import atexit
import time
import copy
from packaging.version import Version
import os.path as osp
import hashlib
from typing import List
import pickle
from collections import OrderedDict
import socket
from threading import Thread, Event
from queue import Queue as SQueue

from collections.abc import Iterable
import concurrent
import gc
import asyncio
from functools import partial
import numpy as np
import torch
import dill
import torch.backends
from .hook_tensor import (profile_tensor_factory,
                        OffloadProfile, TorchOPProfile,
                        keep_funcs)
from .recursive_pickle_obj import recur_dump_obj
from ._utils import iterate_tensor, iterate_all_close, log_dur
from .test_asyncio import AsyncTCPMessageStream
from .schedule import PCI_scheduler
from .random_exec import *
from .profile_ops import profile_ops
from .profile_pickle import profile_pickle
from asyncio import StreamReader, StreamWriter, Queue


# None zero GPU Memory occupancy will be observed due to cuda context
# https://discuss.pytorch.org/t/nvidia-smi-does-not-drop-after-empty-cache-although-cuda-memory-summary-says-there-is-no-current-usage/143741
# print(f"{torch.cuda.memory_allocated()} {torch.cuda.max_memory_allocated()}")

empty_preamble = b"0"*int(1024*1024*0.5) # 0.5MB
class ParallelCollaborativeInference:
    def __init__(self, offload=True,  parallel_approach = "select",
                ip="127.0.0.1", port=12345, ctrl_port=12346, debug=False,
                constraint_latency=False, log=print) -> None:
        self.offload = offload
        self.log = log
        self.server_ip = ip
        self.server_port = port
        self.server_ctrl_port = int(port) + 1
        self.debug = debug
        self.init_forward_count = 0
        self.parallel_approach = parallel_approach
        self.constraint_latency = constraint_latency
        self.fixed_bw = None
        self.select_feature_dims = slice(0)
        self.data_range = 1.0
        self.mvs_block_size = 4
        self.min_stride = 32

        log("Configuring torch to use deterministic behaviors.")
        torch.manual_seed(0)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True

        log(f"parallel approach {parallel_approach}")
        log(f"constraint_latency {constraint_latency}")

    def offload_order(self, bw: int):
        self.fixed_bw = bw

    def start_client(self, model: torch.nn.Module, init_forward_count=0, data_range=1.0, select_feature_dims = None, block_size=4, min_stride=32):
        self.select_feature_dims = select_feature_dims
        self.data_range = float(data_range)
        self.min_stride = min_stride
        self.mvs_block_size = int(block_size)
        assert init_forward_count >= 0
        warned = False
        for p in model.parameters():
            if p.requires_grad and not warned:
                warned = True
                self.log("Warning: model still requires grad. Setting requires grad to False.")
            p.requires_grad = False
        monitor_queue = SQueue()
        Thread(target=asyncio.run,
            args=[self._start_client(model, monitor_queue, init_forward_count)],
            daemon=True).start()
        assert monitor_queue.get() == "Started"

    async def _start_client(self, model: torch.nn.Module,
                        monitor_queue: Queue,
                        init_forward_count=0):
        loop = asyncio.get_event_loop()
        torch.set_grad_enabled(False)
        self.log(f"Connecting to server {self.server_ip}:{self.server_port}")
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.server_ip, self.server_port)
                break
            except (ConnectionError, OSError):
                self.log("Connection error. Retrying.")
                await asyncio.sleep(2)
        ctrl_reader, ctrl_writer = await asyncio.open_connection(
            self.server_ip, self.server_ctrl_port)
        self.log(f"Connected to server {self.server_ip}: {self.server_port}")
        param_num = 0
        for p in model.parameters():
            param_num += p.numel()
        self.log(f"Model parameter number {param_num/1e6:.4f}M.")

        sock = AsyncTCPMessageStream((reader, writer), (ctrl_reader, ctrl_writer), self.log)
        await sock.send_msg(recur_dump_obj(model))
        self.log(f"Send model to server {sock.send_record[0][0]/1024/1024:.4f}MB.")
        await sock.send_obj(init_forward_count)
        input_queue = Queue()
        output_queue = Queue()
        def wrap_forward(*args, **kwargs):
            asyncio.run_coroutine_threadsafe(input_queue.put([args, kwargs]), loop)
            try:
                ret = asyncio.run_coroutine_threadsafe(output_queue.get(), loop).result()
                if ret is None:
                    sys.exit(0)
                return ret
            except (KeyboardInterrupt, SystemExit, concurrent.futures._base.CancelledError):
                self.log("client: terminating...")
                sys.exit(0)
            except Exception as e:
                self.log(str(e))
                raise e
        old_forward = model.forward
        model.forward = wrap_forward
        gc.collect()
        torch.cuda.empty_cache()
        monitor_queue.put("Started")
        self.log("client: started.")
        def close():
            sock.close()
            gc.collect()
            torch.cuda.empty_cache()
            self.log("client: terminated.")
        atexit.register(close)
        try:
            await self.start_common_loop(old_forward,
                                        input_queue, output_queue,
                                        sock, init_forward_count,
                                        role="client", prefix="client: ", log=self.log)
        except (EOFError, KeyboardInterrupt, SystemExit):
            self.log("client: terminating...")
            monitor_queue.put("Failed")
            if output_queue.empty():
                output_queue.put_nowait(None)
            sys.exit(0)
        except Exception as e:
            self.log(str(e))
            raise e

    def start_server(self):
        self.log("starting ParallelCollaborativeInference server...")
        torch.set_grad_enabled(False)
        reader_writers = {}
        streams: List[AsyncTCPMessageStream] = []
        async def _accept(reader: StreamReader, writer: StreamWriter):
            sock: socket.socket = writer.get_extra_info("socket")
            reader_writers[sock.getpeername()[0]] = [reader, writer]
        async def _serve(ctrl_reader: StreamReader, ctrl_writer: StreamWriter):
            sock: socket.socket = ctrl_writer.get_extra_info("socket")
            peername = sock.getpeername()
            prefix = f"server for {peername}: "
            reader, writer = reader_writers[peername[0]]
            stream = AsyncTCPMessageStream((reader, writer), (ctrl_reader, ctrl_writer), self.log)
            streams.append(stream)
            del reader_writers[peername[0]]

            self.log(prefix + "client connected.")
            try:
                model: torch.nn.Module = dill.loads(await stream.recv_msg())
                num_param = int(sum([p.numel() for p in model.parameters()])/1024/1024)
                model_name = f"{model.__class__.__name__}_{num_param}M_{peername[0]}"
                self.log(prefix + f"model {model_name} initial complete.")
                init_forward_count = await stream.recv_obj()
                await self.start_common_loop(model.forward, None, None,
                                        stream, init_forward_count, role="server", prefix=prefix, log=self.log, model_hash=model_name)
            except EOFError:
                self.log(prefix + "terminating server...")
            except Exception as e:
                self.log(e)
                raise e

            stream.close()
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            self.log(prefix + "terminated.")

        async def accept():
            server = await asyncio.start_server(
                _accept, self.server_ip, self.server_port)
            print(f'Serving on {",".join(str(sock.getsockname()) for sock in server.sockets)}')

            async with server:
                await server.serve_forever()
        async def serve():
            server = await asyncio.start_server(
                _serve, self.server_ip, self.server_ctrl_port)
            print(f'Ctrl on {",".join(str(sock.getsockname()) for sock in server.sockets)}')

            async with server:
                await server.serve_forever()
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(asyncio.gather(accept(), serve()))
        except (SystemExit, KeyboardInterrupt):
            for stream in streams:
                stream.close()
            loop.stop()
            self.log("Stopped server.")

    @torch.no_grad()
    async def start_common_loop(self, old_forward, input_queue: Queue, output_queue: Queue,
                            sock: AsyncTCPMessageStream,
                            init_forward_count=0, role="client", prefix="client", log=print, model_hash=""):
        profile_result = OffloadProfile()
        if not self.offload:
            self.parallel_approach = "select"
        if role == "client":
            parallel_approach = self.parallel_approach
            # scheduler = PCI_scheduler(parallel_approach)
            scheduler = PCI_scheduler("mixed2")
            await sock.send_obj(parallel_approach)
        else:
            parallel_approach = await sock.recv_obj()
            # scheduler = PCI_scheduler(parallel_approach)
            self.parallel_approach = parallel_approach
            scheduler = PCI_scheduler("mixed2")

        @torch.no_grad()
        def _profile_forward(*args, **kwargs):
            # change cls before calculating id of the tensors
            hook_level = [0]
            profile_tensor_cls = profile_tensor_factory(profile_result, hook_level)
            args, kwargs = iterate_tensor([args, kwargs], profile_tensor_cls)
            hook_level[0] = 0   # Clear hook level
            profile_result.idx = 0

            # profile input
            args, kwargs = profile_tensor_cls.__torch_function__(None, None, args, kwargs)
            ret = old_forward(*args, **kwargs)
            # profile output
            ret = profile_tensor_cls.__torch_function__(None, None, [ret], None)

            ret = iterate_tensor(ret, torch.Tensor)
            orig_ops_num = len(profile_result.profile)
            self.parse_profile(profile_result)
            profile_pickle(profile_result, log=log)
            profile_result.end_idx = len(profile_result.profile) - 1
            return ret, orig_ops_num

        @torch.no_grad()
        async def profile_forward(warmup=2, repeat=3):
            assert warmup > 0 and repeat > 0
            if role == "client":
                args, kwargs = await input_queue.get()
                await sock.send_obj([args, kwargs])
                log(prefix + "send init input to server")
            else:
                args, kwargs = await sock.recv_obj()
                log(prefix + "recv init input from client")
                time.sleep(10)
            origin_input = copy.deepcopy([args, kwargs])
            log(f"Input size {len(pickle.dumps([args, kwargs]))/1024/1024:.4f}MB")
            log(f"Forwarding for {init_forward_count}(+{warmup} warmup and {repeat} repeat) times for initialization.")
            count = 0
            while count != warmup:
                orig_ret = old_forward(*args, **kwargs)
                torch.cuda.synchronize()
                count += 1

            count = 0
            _init_forward_count = init_forward_count + repeat
            stime = time.time()
            while count != _init_forward_count:
                orig_ret = old_forward(*args, **kwargs)
                torch.cuda.synchronize()
                count += 1
            dur = (time.time() - stime)/count   # Average duration for each forward
            log(f"Forward of the original model takes average {dur:.4f}s.")
            ret1, orig_ops_num = _profile_forward(*origin_input[0], **origin_input[1])
            assert iterate_all_close(orig_ret, ret1)

            log(f"Output size {len(pickle.dumps(ret1))/1024/1024:.4f}MB")

            num = 5
            stime = time.time()
            old_func_args = profile_result.profile[0].func_args
            for _ in range(num):
                profile_result.profile[0].func_args = [args, kwargs]
                ret3 = await local_random_exec(profile_result)
            _dur = (time.time() - stime) / num
            assert iterate_all_close(ret3, ret1)
            log(f"Local random exec takes average {_dur:.4f}s.")

            try:
                log("Using torch.profiler for op profile")
                ret2 = await profile_ops(profile_result, local_random_exec, [profile_result], {})
            except StopIteration:
                log(f"Warning: torch.profiler failed. torch version {torch.__version__}. Falling back to torch.cuda.synchronize based op profile.")
                ret2 = local_random_exec_profile(profile_result, log)

            profile_result.profile[0].func_args = old_func_args
            factor = dur / sum(p.ops_time for p in profile_result.profile.values())
            if role == "client":
                factor += 0.05

            if role == "client" and self.server_ip == "127.0.0.1":    # For debug
                factor = 20.
                profile_result.local_comp_time *= factor
            log(f"Operator records (align ops time with factor {factor:.4f}): ")
            accumulated_time = 0.
            for p in profile_result.profile.values():
                p.ops_time *= factor
                accumulated_time += p.ops_time
                log(f"{p} accu_time {accumulated_time:.4f}s")
            # try to sleep for every 10ms
            profile_result.local_comp_time = sum(
                p.ops_time for p in profile_result.profile.values())
            sleep_step = 100
            log(f"total {len(profile_result.profile)} ops (filtered from {orig_ops_num} ops); time {sum(p.ops_time for p in profile_result.profile.values()):.4f}s (aligned by {factor:.4f} sleep_step {sleep_step}).\n")

            if role == "client":
                await sock.send_obj({"ops": profile_result.copy_for_transmission(), "constraint_latency": self.constraint_latency})
                log("Waiting for graph processing at the server")
                scheduler.recv_plan(await sock.recv_obj())
                log("Got graph plan from server")
            elif role == "server":
                ops_dict = await sock.recv_obj()
                robot_ops = ops_dict["ops"]
                scheduler.update_ops(robot_ops, profile_result)
                constraint_latency = ops_dict["constraint_latency"]
                if "cache" in parallel_approach:
                    scheduler.build_graph(True, parallel_approach)
                else:
                    if "mixed" in parallel_approach:
                        store_plan_path = f"mixed_plan_{model_hash}.pkl"
                    if "mixed" not in parallel_approach or not osp.exists(store_plan_path):
                        if constraint_latency:
                            # fix latency requirement to 1Hz
                            scheduler.required_latency = 1.
                            self.log(f"Setting required_latency to {scheduler.required_latency:.4}s.")
                        self.log("Computing plan for client.")
                        scheduler.build_graph()
                        if "mixed" in parallel_approach:
                            with open(store_plan_path, "wb") as f:
                                pickle.dump({"server plan": scheduler.server_plans,
                                        "client plan": scheduler.client_plans}, f)
                    elif "mixed" in parallel_approach:
                        with open(store_plan_path, "rb") as f:
                            stored_plan = pickle.load(f)
                        scheduler.server_plans, scheduler.client_plans = \
                            stored_plan["server plan"], stored_plan["client plan"]
                        self.log(f"Loaded precomputed mixed plan from {store_plan_path}.")

                await sock.send_obj(scheduler.client_plans)
                scheduler.recv_plan(scheduler.server_plans)  # Server does not output
                self.log(f"Number of local ops {scheduler.info.mixed_layers.sum()}")
                self.log(f"Number of global ops {~(scheduler.info.mixed_layers).sum()}")

            self.log(prefix + "init forward complete.")
            if "cache" in self.parallel_approach:
                offload_plan = scheduler.graph_plan[1]
            else:
                offload_plan = scheduler.graph_plan
            for bw, plan in offload_plan.items():
                to_offload = np.nonzero(plan["offload"])
                to_recv = np.nonzero(plan["recv"])
                est_time = plan["est_time"]
                plan["send_num"] = sum(plan["offload"])
                plan["recv_num"] = sum(plan["recv"])
                # self.log(f"bw {bw}MB/s offload at {to_offload[0].tolist()} recv at {to_recv[0].tolist()} est time {est_time:.4f}s.")

            if role == "client":
                await output_queue.put(ret1)
                await asyncio.sleep(0.2)
            return ret1


        gc.collect()
        torch.cuda.empty_cache()
        await asyncio.sleep(0.)
        if not self.offload:
            while True:
                args, kwargs = await input_queue.get()
                with log_dur(log, prefix=prefix + f"{count} th inference; est bw {0.}MBps"):
                    ret = old_forward(*args, **kwargs)
                    torch.cuda.synchronize()
                    await output_queue.put(ret)
                count += 1
        elif parallel_approach == "all":
            sock.start_send_loop()
            sock.start_recv_loop()
            count = 0
            sock.fix_init_sock_log(prefix + "Init: ")
            while True:
                sock.start_record_log(prefix + f"{count} th sock: ", 1, 1)
                if role == "client":
                    inp = await input_queue.get()
                    with log_dur(sock.add_suffix_to_log, prefix=prefix + f"{count} th inference"):
                        await sock.queued_send(inp)
                        ret = await sock.queued_recv()
                        await output_queue.put(ret)
                else:
                    args, kwargs = await sock.queued_recv()
                    with log_dur(sock.add_suffix_to_log, prefix=prefix + f"{count} th inference"):
                        ret = old_forward(*args, **kwargs)
                        await sock.queued_send(ret)
                count += 1
        elif "cache" in self.parallel_approach:
            try:
                await profile_forward()
            except Exception as e:
                log(str(e))
                raise e
            cache_plan = scheduler.graph_plan[0]
            offload_plan = scheduler.graph_plan[1]

            sock.start_send_loop()
            sock.start_recv_loop()

            count = 0
            # profile_result.profile[0].func_args = [None, None]
            sock.fix_init_sock_log(prefix + "Init: ")
            from .cached_exec import CacheInfo, robot_cached_inference, server_cached_inference
            if role == "client":
                from .cached_schedule import look_ahead_step, ROBOT, SERVER
                from .look_ahead import look_ahead_process
                from multiprocessing import Queue, Process
                obs_queue = Queue()
                order_queue = Queue()
                predict_queue = Queue()
                Process(target=look_ahead_process, args=(
                    look_ahead_step, order_queue, obs_queue, predict_queue),
                        name="look_ahead_process", daemon=True).start()

            cache_info = CacheInfo()
            cache_info.feature_slice = self.select_feature_dims
            cache_info.mvs_block_size = self.mvs_block_size
            cache_info.min_stride = self.min_stride
            local = False
            if "local" in self.parallel_approach:
                for key, val in cache_plan.items():
                    cache_plan[key][-1] = ROBOT
                cache_info.cache_location = ROBOT
                log("Setting all location to be on the robot.")
                local = True
            if "eva2" in self.parallel_approach:
                update = False
                if "high" in self.parallel_approach:
                    cache_info.psnr_threshold = 30.
                    log("Setting a high psnr threshold.")
                else:
                    cache_info.psnr_threshold = 10.
                    log("Setting a low psnr threshold.")
            else:
                update = True
                cache_info.psnr_threshold = 20.

            if role == "client":
                cache_info.data_range = self.data_range
                cache_info.key = tuple([scheduler.max_bw, 0, 0, ROBOT])
                cache_info.cache_op, cache_info.cache_location = cache_plan[tuple([scheduler.max_bw, 0, 0, ROBOT])]
                _p = profile_result.profile[cache_info.cache_op]
                p = profile_result.profile[_p.input_from[0]]
                local_dim = p.local_dim
                cache_info.cached_input_image_height, cache_info.cached_input_image_width = profile_result.profile[0].output_shapes[0][-2:]
                cache_info.local_dim = local_dim
                cache_info.stride, cache_info.offset, cache_info.field_len = p.stride, p.offset, p.field_len
                cache_info.cached_activations_height, cache_info.cached_activations_width =\
                    p.output_shapes[0][local_dim-1:local_dim+1]
                cache_info.flatten_orig_shape = list(p.output_shapes[0][1:local_dim-1]) + list(
                    p.output_shapes[0][local_dim+1:]
                )
                cache_info.fixed_cache_op = cache_info.cache_op
                cache_info.block_num_to_x = cache_info.mvs_block_size ** 2 / cache_info.cached_input_image_height / cache_info.cached_input_image_width
                await sock.queued_send(cache_info)
                await asyncio.sleep(0.2)
            else:
                cache_info = await sock.queued_recv()

            while True:
                if role == "client":
                    profile_result.profile[0].func_args = await input_queue.get()
                    profile_result.profile[0].func_args = iterate_tensor(
                        profile_result.profile[0].func_args, lambda x:x)
                # TODO sock.start_record_log
                sock.start_record_log(prefix + f"{count} th sock: ")
                with log_dur(sock.add_suffix_to_log, prefix=prefix + 
                                f"{count} th inference"):
                    if role == "client":
                        await robot_cached_inference(count, profile_result, sock, cache_info, cache_plan, offload_plan, order_queue, obs_queue, predict_queue, update, local)
                    else:
                        await server_cached_inference(count, profile_result, sock, cache_info, cache_plan, offload_plan, update, local)
                if role == "client":
                    await output_queue.put(profile_result.ret_store)
                count += 1
        else:   # HP
            try:
                await profile_forward()
            except Exception as e:
                log(str(e))
                raise e
            offload_plan = scheduler.graph_plan
            sock.start_send_loop()
            sock.start_recv_loop()

            count = 0
            profile_result.profile[0].func_args = [None, None]
            sock.fix_init_sock_log(prefix + "Init: ")

            if self.offload:
                while True:
                    if role == "client":
                        profile_result.profile[0].func_args = await input_queue.get()
                        last_bandwidth = int(min(sock.last_bandwidth, scheduler.max_bw)) \
                                if self.fixed_bw is None else self.fixed_bw
                        await sock.queued_send(last_bandwidth)
                        plan = offload_plan[last_bandwidth]
                        sock.start_record_log(prefix + f"{count} th sock: ",
                                            plan["send_num"] + 1,   # bw, send_num
                                            max(plan["recv_num"], 1))   # recv_num / preamble bw
                    else:
                        last_bandwidth = await sock.queued_recv()
                        plan = offload_plan[last_bandwidth]
                        sock.start_record_log(prefix + f"{count} th sock: ",
                                            max(plan["send_num"], 1),   # send_num / preamble bw
                                            plan["recv_num"] + 1)   # recv_num + bw
                    with log_dur(sock.add_suffix_to_log, prefix=prefix + f"{count} th inference est bw {last_bandwidth}MBps, est exec time {plan['est_time']:.4f}s"):
                        if plan["recv_num"] == 0:
                            if role == "client":
                                sock.skip_next_recv()
                            else:
                                if sock.sending:    # Last sock send msg unfinished
                                    await sock.queued_send(b"")
                                else:
                                    await sock.queued_send(empty_preamble)
                                count += 1
                                continue
                        # await random_exec_filtered(profile_result.exec_plan[last_bandwidth], sock, sock.add_suffix_to_log, **plan)
                        # await random_exec(profile_result, sock, sock.add_suffix_to_log, **plan)
                        await random_exec_compiled(profile_result, last_bandwidth)
                        torch.cuda.synchronize()
                    if role == "client":
                        await output_queue.put(profile_result.ret_store)
                    count += 1
            else:
                while True:
                    args, kwargs = await input_queue.get()
                    with log_dur(log, prefix=prefix + f"{count} th inference; est bw {0.}MBps, est exec time {profile_result.local_comp_time:.4f}s; "):
                        ret = old_forward(*args, **kwargs)
                        torch.cuda.synchronize()
                        await output_queue.put(ret)
                    count += 1
    

    def parse_profile(self, profile_result: OffloadProfile):
        all_profiles = profile_result.profile
        idx_array = list(all_profiles.keys())
        for idx in idx_array:
            profile: TorchOPProfile = all_profiles[idx]
            input_ids: list = profile.input_ids
            output_ids: list = profile.output_ids

            # parse input/output relationship by querying id in previous output
            for i, _id in enumerate(input_ids):
                for _idx in range(0, idx):
                    if _id in all_profiles[_idx].output_ids:
                        hit_idx = all_profiles[_idx].output_ids.index(_id)
                        if _idx in profile.input_from:
                            self.log(f"Warning: {idx}op has duplicated input from {_idx}op")
                        else:
                            profile.input_from.append(_idx)
                        if idx in all_profiles[_idx].output_to:
                            self.log(f"Warning: {_idx}op has duplicated output to {idx}op")
                        else:
                            all_profiles[_idx].output_to.append(idx)
                        if hit_idx not in all_profiles[_idx].output_idx_slots:
                            all_profiles[_idx].output_idx_slots[hit_idx] = [
                                profile.input_slots[i]]
                            # {output_idx: [(op_idx, input_idx)]}
                        else:
                            all_profiles[_idx].output_idx_slots[hit_idx].append(
                                profile.input_slots[i])

            # Since id can be reused, remove any duplicated id in previous output
            for _id in output_ids:
                for _idx in range(0, idx):
                    if _id in all_profiles[_idx].output_ids:
                        hit_idx = all_profiles[_idx].output_ids.index(_id)
                        all_profiles[_idx].output_ids[hit_idx] = None


        for idx in reversed(idx_array):    # ignore end
            profile = all_profiles[idx]
            output_idx_slots = profile.output_idx_slots
            if len(output_idx_slots) > 1:
                # sort keys of ordered dict in ascending order
                sorted_output_idx_slots = OrderedDict()
                sorted_keys = sorted(list(output_idx_slots.keys()))
                for _idx in sorted_keys:
                    sorted_output_idx_slots[_idx] = output_idx_slots[_idx]
                profile.output_idx_slots = sorted_output_idx_slots
            if len(profile.output_to) == 0 and not profile.keep:
                # if no output and not explicitly keep, remove this profile
                for i, _idx in enumerate(profile.input_from):
                    all_profiles[_idx].output_to.remove(idx)
                    # Remove output slots that fills the input of the current profile
                    for key, slots in all_profiles[_idx].output_idx_slots.items():
                        remain_slots = []
                        for slot in slots:
                            if slot.idx != idx:
                                remain_slots.append(slot)
                        all_profiles[_idx].output_idx_slots[key] = remain_slots
                for _idx in range(idx+1, idx_array[-1]+1):
                    if _idx in all_profiles:
                        _profile = all_profiles[_idx]
                        _profile.idx -= 1
                for _profile in all_profiles.values():
                    for i, _ in enumerate(_profile.output_to):
                        if _profile.output_to[i] > idx:
                            _profile.output_to[i] -= 1
                    for i, _ in enumerate(_profile.input_from):
                        if _profile.input_from[i] > idx:
                            _profile.input_from[i] -= 1
                del all_profiles[idx]
        profile_result.profile = OrderedDict()
        for i, profile in enumerate(all_profiles.values()):
            for slot in profile.input_slots:
                slot.idx = i
            profile_result.profile[i] = profile

        # patch __setitem__: __setitem__ does not have a return value
        # but only modifies input_from[0] inplace;
        # Correct the data dependency here and also __setitem__ should not be offloaded.
        for i, profile in enumerate(profile_result.profile.values()):
            if profile.func_name == "__setitem__":
                inplace_mod_idx = profile.input_from[0]
                for idx in profile_result.profile[inplace_mod_idx].output_to:
                    if idx > i:     # op after this inplace __setitem__ also depends on this op
                        _p = profile_result.profile[idx]
                        _p.input_from.append(i)
                        profile.output_to.append(idx)

        # Check profile valid
        new_all_profiles = profile_result.profile
        for key, profile in new_all_profiles.items():
            assert key == profile.idx
            assert len(profile.output_to) > 0 or profile.func_name in keep_funcs
            if not len(profile.output_idx_slots) == len(profile.output_shapes) and profile.func_name != "_end":
                raise RuntimeError(str(profile))
            for idx in profile.output_to:
                assert idx in new_all_profiles
                assert key in new_all_profiles[idx].input_from
            for idx in profile.input_from:
                assert idx in new_all_profiles
                assert key in new_all_profiles[idx].output_to

        end_profile = new_all_profiles[len(new_all_profiles)-1]
        for i, ret_slot in enumerate(profile_result.ret_slots):
            ret_slot.idx = end_profile.idx
            end_profile.output_idx_slots[i] = [ret_slot]
            end_profile.output_to = [-1] * len(profile_result.ret_slots)

        # Temp fix for branches
        idx_array = list(new_all_profiles.keys())
        for idx in idx_array:
            profile = new_all_profiles[idx]
            valid_output_len = len(profile.output_to)
            if valid_output_len > 1:
                current_end = set(profile.output_to)
                while len(current_end) > 1:
                    current_idx = min(current_end)
                    current_end = list(current_end)
                    del current_end[current_end.index(current_idx)]
                    current_end += new_all_profiles[current_idx].output_to
                    current_end = set(current_end)

                for _idx in range(idx, list(current_end)[0] + 1):
                    new_all_profiles[_idx].masked = True
        new_all_profiles[idx_array[-1]].masked = False

        profile_result.profile[0].local_dim = len(profile_result.profile[0].output_shapes[0]) - 1
        profiles = profile_result.profile
        for profile in profile_result.profile.values():
            idx: int = profile.idx
            func: str = profile.func_name
            func_args = profile.func_args
            input_shapes = profile.input_shapes
            output_shapes = profile.output_shapes
            if idx > 0:
                parent_profile = profile_result.profile[profile.input_from[0]]
                last_local_dim = parent_profile.local_dim
            else:
                parent_profile = None
                last_local_dim = len(profile.input_shapes[0]) - 1
            if idx in [profile_result.end_idx, 0]:
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": True, "profile": profile}
                if parent_profile:
                    profile.local_dim = parent_profile.local_dim
            elif func in ["__getitem__"]:
                args, kwargs = func_args
                slice_arg: list = args[1]
                assert len(kwargs) == 0
                if last_local_dim is None:
                    barrier = True
                    profile.local_dim = None
                elif isinstance(slice_arg, int): # get item at the first dim
                    if slice_arg == last_local_dim:
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False
                        profile.local_dim = last_local_dim - 1
                elif isinstance(slice_arg, (torch.Tensor)):
                    # TODO: this situation is very complicated to analyse, leave it to future
                    if len(input_shapes[0]) != len(output_shapes[0]):
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False     
                        profile.local_dim = last_local_dim
                elif isinstance(slice_arg, list):
                    slice_arg = list(slice_arg)
                    try:
                        find_ellipsis = slice_arg.index(...)
                    except ValueError:
                        find_ellipsis = None
                    len_diff = len(input_shapes[0])-len(slice_arg)
                    if find_ellipsis is not None:
                        origin_shape_len = len(input_shapes[0])
                        ellipsis_len = origin_shape_len - len(slice_arg)
                        ellipsis_idx = slice_arg.index(...)
                        [slice_arg.insert(ellipsis_idx, None) for _ in range(ellipsis_len)]
                    elif len_diff > 0:
                        slice_arg += len_diff * [None]
                    if isinstance(slice_arg[last_local_dim], int):
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False
                        dim_reduced = 0
                        for i, a in enumerate(slice_arg):
                            if i == last_local_dim or i == last_local_dim - 1:
                                if isinstance(a, slice) and a.step is not None:
                                    if parent_profile:
                                        profile.stride = parent_profile.stride * a.step
                                    else:
                                        profile.stride = a.step
                            if i == last_local_dim:
                                profile.local_dim = last_local_dim - dim_reduced
                                break
                            if isinstance(a, int):
                                dim_reduced += 1
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["__setitem__"]:
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
                profile.local_dim = parent_profile.local_dim
            elif func.startswith(("cat")):
                barrier = last_local_dim is None
                args, kwargs = func_args
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = 0
                if dim != last_local_dim and len(profile.input_shapes) > len(args[0]):
                    align_shape = True
                else:
                    align_shape = False
                profile.align_shape = align_shape
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile,
                    "apply_rate": True, "align_shape": align_shape} # Regular offloading
                profile.local_dim = parent_profile.local_dim
            elif func.startswith((
                "add", "sub", "rsub", "div", "mul",
                "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
                "exp", "pow",
                )) or func in [
                    "le", "lt", "gt", "ge", "eq","nms",
                ]: # Element-wise operations that keep tensor shape
                align_shape = False
                if last_local_dim is not None and len(input_shapes) < 2:
                    mat = []
                    for arg in func_args[0]:
                        if isinstance(arg, torch.Tensor) and len(arg.shape) > last_local_dim and arg.shape[last_local_dim] > 1:
                            mat.append(True)
                        else:
                            mat.append(False)
                    align_shape = np.all(mat)
                profile.align_shape = align_shape
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile, "align_shape": align_shape} # Regular offloading
                profile.local_dim = parent_profile.local_dim
            elif func.startswith(("sin", "cos", "tan", "asin", "acos", "atan", "arc",
                "batch_norm", "layer_norm",
                "relu", "rrelu", "gelu", "sigmoid", "sign", "selu", "hardswish",
                "hardsigmoid", "silu", 
                "sqrt", "rsqrt",)) or func in [
                    "contiguous", "interpolate", "clone", "detach", 
                    "float", "int", "double", "long", "abs", "type"]:
                if func == "interpolate":
                    scale_factor = func_args[1]["scale_factor"]
                    profile.local_stride = 1/scale_factor
                    profile.stride /= scale_factor  # TODO
                    profile.interpolated = scale_factor
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
                profile.local_dim = parent_profile.local_dim
            elif func in ["view", "reshape"]:
                barrier = False
                if last_local_dim is not None:
                    input_shape = input_shapes[0]
                    output_shape = output_shapes[0]
                    search_size_h = np.prod(input_shape[:last_local_dim])
                    search_size_hw = np.prod(input_shape[:last_local_dim+1])
                    cum_shape = np.cumprod(output_shape)
                    hw_remained = np.any(cum_shape == search_size_h) and np.any(cum_shape == search_size_hw)
                    if not hw_remained:
                        barrier = True
                    searched = np.nonzero(cum_shape == search_size_hw)[0]
                    if len(searched):
                        if not barrier:
                            profile.local_dim = searched[0]
                        args, kwargs = func_args
                        searched_idx = searched[0]
                        if -1 in args[1:] and args[1+searched_idx] != -1:
                            idx = args[1:].index(-1)
                            args[1+idx] = output_shape[idx]
                        args[1+searched_idx] = -1   # Change the shape of local dim to be flexible
                    else:
                        profile.local_dim = None
                        barrier = True
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["flatten", "ravel"]:
                if last_local_dim is not None and last_local_dim == len(input_shapes[0])-1:
                    profile.local_dim = 0
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["unflatten"]:
                args, kwargs = func_args
                dim = args[1]
                if dim != last_local_dim:
                    if dim < last_local_dim:
                        profile.local_dim = last_local_dim + len(args[2]) - 1
                    else:
                        profile.local_dim = last_local_dim
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["squeeze", "unsqueeze"]:
                barrier = False
                if last_local_dim is not None:
                    input_shape = input_shapes[0]
                    output_shape = output_shapes[0]
                    search_size = np.prod(input_shape[:last_local_dim+1])
                    searched = np.nonzero(np.cumprod(output_shape) == search_size)[0]
                    if len(searched):
                        profile.local_dim = searched[0]
                    else:
                        profile.local_dim = None
                        barrier = True
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["unbind"]:
                args, kwargs = func_args
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = 0
                if last_local_dim is not None and dim != last_local_dim:
                    barrier = False
                    if dim > last_local_dim:
                        profile.local_dim = last_local_dim
                    else:
                        profile.local_dim = last_local_dim - 1
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["max", "min", "any", "all", "argmax", "argmin"]:
                args, kwargs = func_args
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = None
                if last_local_dim is not None and dim is not None and dim != last_local_dim:
                    barrier = False
                    if "keepdim" in kwargs and kwargs["keepdim"] or len(args) > 2 and args[2] or dim > last_local_dim:
                        profile.local_dim = last_local_dim
                    else:
                        profile.local_dim = last_local_dim - 1
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func.startswith(("permute")):
                if last_local_dim is not None:
                    barrier = False
                    args, _ = func_args
                    if len(args) == 2 and isinstance(args[1], list):
                        indices = args[1]
                    else:
                        indices = func_args[0][1:]
                    searched = np.nonzero(
                        np.arange(len(input_shapes[0]))[indices] == last_local_dim)[0]
                    if len(searched):
                        profile.local_dim = searched[0]
                    else:
                        profile.local_dim = last_local_dim
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func.startswith((
                "conv_transpose2d"
            )): # Convolution operations
                profile.hook_kwargs = {
                    "idx": idx, "conv": True, "unconv": True,
                    "barrier": False, "profile": profile}
                profile.local_dim = len(input_shapes[0]) - 1
            elif func.startswith((
                "conv", "max_pool", "avg_pool",
            )): # Convolution operations
                profile.hook_kwargs = {
                    "idx": idx, "conv": True,
                    "barrier": False, "profile": profile}
                profile.local_dim = len(input_shapes[0]) - 1
            elif func.startswith((
                "bmm",
            )): # Convolution operations
                assert len(input_shapes) == 2, str(profile)
                if profile_result.profile[profile.input_from[0]].local_dim != len(input_shapes[0]) - 1 and \
                    profile_result.profile[profile.input_from[1]].local_dim != len(input_shapes[1]) - 2:
                    profile.local_dim = last_local_dim
                    barrier = False
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func.startswith((
                "softmax",
            )):
                args, kwargs = func_args
                if len(args) > 1:
                    dim = args[1]
                else:
                    dim = kwargs["dim"]
                if dim == -1:
                    dim = len(input_shapes[0]) - 1
                if last_local_dim is not None and dim != last_local_dim:
                    profile.local_dim = last_local_dim
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func.startswith(("linear")):
                # linear only applies to the last dim; if local dim is not the last dim, the locality remains
                if last_local_dim is not None and last_local_dim != len(input_shapes[0]) - 1:
                    barrier = False
                    profile.local_dim = last_local_dim
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func in ["shape", "dim"]:
                raise RuntimeError
            else:
                # Operation that does not support offloading.
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": True, "profile": profile}
                profile.local_dim = None   # Operations that destroyed locality

            # if profile.local_dim is None or not parent_profile.caching or profile.masked:
            #     profile.caching = False

            if func.startswith(("conv", "max_pool", "avg_pool")):
                profile.interpolated *= max(profiles[i].interpolated for i in profile.input_from)
                assert parent_profile is not None
                last_offset = parent_profile.offset
                last_stride = parent_profile.stride
                last_temp_field_len = parent_profile.temp_field_len
                args, kwargs = func_args
                if func.startswith(("conv")):
                    if "stride" in kwargs:
                        stride = kwargs["stride"]
                    else:
                        stride = args[3]
                    if "padding" in kwargs:
                        padding = kwargs["padding"]
                    else:
                        padding = args[4]
                    kernel_size = args[1].shape[-1]
                elif func.startswith(("max_pool", "avg_pool")):
                    if "stride" in kwargs:
                        stride = kwargs["stride"]
                    else:
                        stride = args[2]
                    if "padding" in kwargs:
                        padding = kwargs["padding"]
                    else:
                        padding = args[3]
                    if "kernel_size" in kwargs:
                        kernel_size = kwargs["kernel_size"]
                    else:
                        kernel_size = args[1]
                else:
                    raise RuntimeError(str(profile))
                if isinstance(kernel_size, Iterable):
                    kernel_size = list(kernel_size)[-1]
                if isinstance(stride, Iterable):
                    stride = list(stride)[-1]
                if isinstance(padding, Iterable):
                    padding = list(padding)[-1]
                # x * stride - offset = img_x
                # last_x * last_stride - last_offset = img_x
                # last_end_x * last_stride+ last_temp_field_len = img_end_x + last_offset
                if "transpose" not in func: # normal conv
                    if last_stride > 0:
                        offset = padding * last_stride + last_offset
                    else:
                        offset = padding
                    if last_stride == 0:
                        temp_field_len = kernel_size
                    elif last_temp_field_len == 0:
                        temp_field_len = kernel_size * last_stride
                    else:
                        temp_field_len = (kernel_size - 1) * last_stride + last_temp_field_len
                    if last_stride == 0:
                        _stride = stride
                    else:
                        _stride = stride * last_stride
                else:   # unconv
                    assert last_offset > 0 and last_temp_field_len > 0 and last_stride > 0
                    offset = last_offset - padding * last_stride
                    temp_field_len = last_temp_field_len - (kernel_size - 1) * last_stride
                    _stride = last_stride // stride 
                    assert last_stride % stride  == 0
                profile.kernel_size, profile.local_stride, profile.padding = kernel_size, stride, padding
                output_h, output_w = profile.output_shapes[0][-2:]
                input_image_h, input_image_w = profile_result.profile[0].output_shapes[0][-2:]

                # assert (output_h - 1) * _stride + temp_field_len <= input_image_h + offset * 2 and output_h * _stride + temp_field_len >= input_image_h + offset * 2, \
                #     f"Error: activation_h {output_h} offset {offset} stride {_stride} temp_field_len {temp_field_len}; input_image_h {input_image_h}" 
                # assert (output_w - 1) * _stride + temp_field_len <= input_image_w + offset * 2 and output_w* _stride + temp_field_len >= input_image_w + offset * 2, \
                #     f"Error: activation_h {output_w} offset {offset} stride {_stride} temp_field_len {temp_field_len}; input_image_h {input_image_w}"

                profile.offset, profile.stride, profile.temp_field_len = offset, _stride, temp_field_len
                if profile.interpolated > 1:
                    profile.field_len = int(profile.interpolated * profile.stride)
                else:
                    profile.field_len = temp_field_len
            else:
                if parent_profile:
                    profile.offset = max(profile.offset,
                                        min([profiles[i].offset for i in profile.input_from]))
                    profile.stride = max(profile.stride,
                                        min([profiles[i].stride for i in profile.input_from]))
                    profile.temp_field_len = max(profile.temp_field_len,
                                        min([profiles[i].temp_field_len for i in profile.input_from]))

                    profile.interpolated *= max(profiles[i].interpolated for i in profile.input_from)
                    if profile.interpolated > 1:
                        profile.field_len = int(profile.interpolated * profile.stride)
                    else:
                        profile.field_len = profile.temp_field_len
                        # for i in profile.input_from:
                        #     if profiles[i].func_name == "interpolate":
                        #         profile.stride = profile.stride
                        #         profile.offset = profile.offset
        # Fix field len
        input_image_h, input_w = profile_result.profile[0].output_shapes[0][-2:]
        
        last_local_operator = max(p.idx for p in profile_result.profile.values() if p.local_dim)
        first_non_local_op = profile_result.profile[last_local_operator].output_to[0]
        # fix input of first_non_local_op
        profile = profile_result.profile[first_non_local_op]
        if len(profile.input_from):
            parent_profile = profile_result.profile[profile.input_from[0]]
            local_dim = parent_profile.local_dim
            if local_dim and len(profile.input_shapes):
                offset, stride, field_len = profile.offset, profile.stride, profile.field_len
                output_h, output_w = profile.input_shapes[0][local_dim-1:local_dim+1]
                log2 = np.log2(field_len)
                need_fix = False
                need_fix = not ((output_h - 1) * stride + field_len <= input_image_h + offset * 2 and 
                                    output_h * stride + field_len > input_image_h + offset * 2 and \
                                (output_w - 1) * stride + field_len <= input_image_w + offset * 2 and
                                    output_w * stride + field_len > input_image_w + offset * 2)
                if log2 % 1 > 0 or need_fix:    # Need to pad
                    if input_image_h == 224:
                        field_len = 32
                    else:
                        field_len = int(2 ** max(np.ceil(log2), 5))
                    offset = int(np.ceil(((output_h - 1) * stride + field_len - input_image_h) / 2))
                    profile.offset, profile.stride, profile.field_len = offset, stride, field_len
                    parent_profile.offset, parent_profile.stride, parent_profile.field_len = offset, stride, field_len
