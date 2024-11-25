import socket
import pickle
import struct
from queue import Queue
from threading import Thread
from typing import Iterable, Union, List, Dict
import time
import torch
import inspect
import numpy as np


MAX_RECV_SIZE = 4*1024*1024
class TCPMessageStream:
    def __init__(self, sock: socket.socket, ctrl_sock: socket.socket):
        self.sock = sock
        self.ctrl_sock = ctrl_sock
        ctrl_sock.settimeout(None)
        sock.settimeout(None)
        self.last_bandwidth = 1000      # send bandwidth
        self.last_recv_bandwidth = 1000
        self.last_recv_took = [0.]             # second
        self.last_send_took = [0.]
        self.header_fmt = 'idd' # [int, double, double]: [msize, send bw, send took]
        self.header_size = struct.calcsize(self.header_fmt)
        self.speed_est_stale = 0
        self.speed_est_stale_thres = 3

    def init_transmission_time_record(self):
        self.last_recv_took = [0.]
        self.last_send_took = [0.]

    def get_transmission_time(self):
        self.last_send_took.append(self.last_send_took[-1])
        self.last_recv_took.append(self.last_recv_took[-1])
        return sum(self.last_send_took + self.last_recv_took)

    def unpack_header(self, header):
        return struct.unpack(self.header_fmt, header)

    def pack_header(self, msg):
        return struct.pack(self.header_fmt,
                           len(msg), self.last_bandwidth, self.last_send_took[-1]) + msg

    def send_msg(self,msg):
        msize = len(msg) + self.header_size
        stime = time.time()
        self.sock.sendall(self.pack_header(msg))
        assert len(self.ctrl_sock.recv(1)) == 1  # Recv one byte to confirm transmission
        send_took = time.time() - stime
        if msize > 1024 * 10:
            self.last_bandwidth = msize / 1024 / 1024 / send_took
            self.last_send_took.append(send_took)
        print(f"send {msize/1024/1024:.3f}MB at bw {self.last_bandwidth:.2f}MB")

    def recv_msg(self):
        buffer = bytearray()
        while len(buffer) < self.header_size:
            buffer += self.sock.recv(MAX_RECV_SIZE)
        msize, last_recv_bandwidth, last_recv_took = self.unpack_header(
            buffer[:self.header_size])  # Last send bandwidth from the opposite
        self.last_recv_took.append(last_recv_took)
        buffer = buffer[self.header_size:]
        while len(buffer) < msize:
            buffer += self.sock.recv(MAX_RECV_SIZE)
        self.ctrl_sock.send(b"0")    # Send one byte to confirm transmission
        self.last_recv_bandwidth = last_recv_bandwidth
        print(f"recv {msize/1024/1024:.3f}MB; last recv bandwidth {self.last_recv_bandwidth:.2f}MB/s.")
        return buffer[:msize]

    def send_obj(self, obj):
        msg = pickle.dumps(obj)
        self.send_msg(msg)

    def recv_obj(self):
        return pickle.loads(self.recv_msg())

    def close(self):
        self.sock.close()
        self.ctrl_sock.close()

    def send_obj_from_queue(self, queue: Queue, num=1):
        """register a thread to send incoming objs from queue"""
        def send():
            try:
                for _ in range(num):
                    self.send_obj(queue.get())
            except (ConnectionResetError, OSError):
                pass
            print("Send thread terminated.")
        Thread(target=send, name="send", daemon=True).start()

    def put_recved_into_queue(self, queue: Queue, num=1):
        """register a thread to put next incoming objs into queue"""
        def put():
            for _ in range(num):
                try:
                    queue.put(self.recv_obj())
                except (ConnectionResetError, OSError):
                    queue.put(None)
        Thread(target=put, name="put", daemon=True).start()

    def get_bandwidth(self):
        return self.last_bandwidth

class EndOfExec(Exception):
    """End of forward execution due to all tensor being transferred."""

def iterate_all_close(obj1, obj2, rtol=1e-5, atol=1e-8):
    tensors1 = []
    tensors2 = []
    iterate_tensor(obj1, tensors1.append)
    iterate_tensor(obj2, tensors2.append)
    assert len(tensors1) == len(tensors2)
    for t1, t2 in zip(tensors1, tensors2):
        if not torch.allclose(t1, t2, rtol, atol):
            return False
    return True

def iterate_tensor(obj, func, cls=torch.Tensor):
    """iterate through items in obj and apply func to tensors among the items.
    Return an new obj with the same structure as the obj.
    """
    if isinstance(obj, cls):
        return func(obj)
    if isinstance(obj, dict):
        _obj = {}
        for key, val in obj.items():
            if isinstance(val, cls):
                _obj[key] = func(val)
            else:
                _obj[key] = iterate_tensor(val, func, cls)
        return _obj
    if isinstance(obj, Iterable) and not isinstance(
        obj, (str, np.ndarray, torch.Size)):
        new_obj = []
        for _obj in obj:
            if isinstance(_obj, cls):
                _obj = func(_obj)
            else:
                _obj = iterate_tensor(_obj, func, cls)
            new_obj.append(_obj)
        return new_obj
    return obj

def iterate_tensor_with_reference(obj, func, cls=torch.Tensor,
                                container: Union[List, Dict]=None,
                                index: Union[int, str]=None):
    """iterate through items in obj and apply func to tensors among the items.
    Also passes the container and index so that the tensor can be access by container[index].
    Return an new obj with the same structure as the obj.
    """
    if isinstance(obj, torch.Size):
        return obj
    if isinstance(obj, cls):
        return func(obj, container, index)
    elif isinstance(obj, dict):
        _obj = {}
        _obj.update(obj)
        for key, val in _obj.items():
            if isinstance(val, cls):
                _obj[key] = func(val, _obj, key)
            else:
                _obj[key] = iterate_tensor_with_reference(
                    val, func, cls, _obj, key)
        return _obj
    elif isinstance(obj, Iterable) and not isinstance(obj, (torch.Tensor, str)):
        new_obj = list(obj)
        for i, _obj in enumerate(new_obj):
            if isinstance(_obj, cls):
                new_obj[i] = func(_obj, new_obj, i)
            else:
                new_obj[i] = iterate_tensor_with_reference(
                    _obj, func, cls, new_obj, i)
        return new_obj
    return obj

def args_kwargs_to_args(func, args, kwargs=None):
    """Turns args and kwargs to args only
    """
    if kwargs is None:
        kwargs = {}
    try:
        signature = inspect.signature(func)
        parameters = signature.bind(*args, **kwargs)
        parameters.apply_defaults()
        last_val = list(signature.parameters.values())[-1]
        if last_val.kind is last_val.VAR_KEYWORD:
            for key in kwargs.keys():
                # If the function is accepting **kwargs and
                # we are passing key word argument to **kwargs;
                # This is very rare in torch coding.
                if key not in parameters.arguments:
                    raise RuntimeError(
                        f"Passing variable key word argument {key} is not supported.")
        elif last_val.kind is last_val.VAR_POSITIONAL and len(parameters.arguments) == 1:
            args = list(parameters.arguments.values())[0]
        else:
            args = list(parameters.arguments.values())
        assert len(args) == len(parameters.arguments)
    except RuntimeError as e:
        if len(kwargs) > 0:
            args = list(args) + list(kwargs.values())
    return args



class LogStamp:
    def __init__(self, log=print, prefix="") -> None:
        self.log = log
        self.stime = 0.
        self.prefix = prefix

    def __enter__(self):
        self.stime = time.time()

    def __exit__(self, *args, **kwargs):
        etime = time.time()
        self.log(self.prefix +
                 f" starts at {self.stime}; ends at {etime}; dur {etime - self.stime:.4f}s")

def log_dur(log=print, prefix=""):
    return LogStamp(log, prefix)
