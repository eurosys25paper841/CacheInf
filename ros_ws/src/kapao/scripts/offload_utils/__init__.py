from queue import Queue, Empty, Full
from threading import Event, Lock
import rospy

from .tracking_utils import *
class LatestQueue:
    '''Thread safe queue enabling deletion of stale obj.
    '''
    def __init__(self, maxsize=1) -> None:
        self.queue = Queue(maxsize=maxsize)
        self.lock = Lock()
        self.is_empty = Event()
        self.is_empty.set()

    def get(self, timeout=2):
        data = self.queue.get()
        if self.queue.empty():
            self.is_empty.set()
        return data

    def put(self, obj):
        with self.lock:
            while not rospy.is_shutdown():
                try:
                    if self.queue.full():
                        self.queue.get_nowait()
                    self.queue.put_nowait(obj)
                    self.is_empty.clear()
                    break
                except (Empty, Full) as _:
                    continue

    def get_nowait(self):
        data = self.queue.get_nowait()
        if self.queue.empty():
            self.is_empty.set()
        return data

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

class LogState:
    def __init__(self, true_msg, false_msg, init_state=False) -> None:
        self.state = init_state
        self.true_msg = true_msg
        self.false_msg = false_msg

    def __call__(self, state):
        if self.state != state:
            if state:
                rospy.loginfo(self.true_msg)
            else:
                rospy.loginfo(self.false_msg)
        self.state = state
        return state