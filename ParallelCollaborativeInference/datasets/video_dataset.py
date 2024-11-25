import os
import os.path as osp
import sys
import cv2
import time
from typing import List
import numpy as np


class VideoFrameDataset:
    def __init__(self, all_sequence_dir, seq_name,
                 log=print, debug=False, frame_rate=30, play_dur=30):
        # infinitely looping forward and backward over a sequence
        self.debug = debug
        self.log = log
        self.path = os.path.join(all_sequence_dir, seq_name)
        self.play_dur = play_dur
        self.frame_dur = 1./frame_rate
        self.sequence_dir: List[str] = []
        for d in sorted(os.listdir(self.path)):
            frame = osp.join(self.path, d)
            self.sequence_dir.append(frame)
        log(f"Dataset {self.path} number of frames {len(self.sequence_dir)}")
        self.frame_idx = 0

    def __len__(self):
        return int(1e8)

    def next_sequence(self):
        # infinite loop
        try:
            self.current_sequence = next(self.sequence_iter)
        except StopIteration:
            self.sequence_iter = iter(self.sequence_dir)
            self.current_sequence = next(self.sequence_iter)
        frames = []
        self.log(f"Current sequence {self.current_sequence}")
        for f in sorted(os.listdir(self.current_sequence)):
            if f.endswith((".jpg", ".png")):
                frames.append(osp.join(self.current_sequence, f))
        self.current_sequence_frames = frames

        self.sequence_start_time = time.time()

    def __getitem__(self, i):
        sequence_len = len(self.sequence_dir)
        forward_backward = (self.frame_idx // sequence_len) % 2
        if forward_backward == 0:   # forward
            true_idx = self.frame_idx % sequence_len
        else:                       # backward
            true_idx = - (self.frame_idx % sequence_len + 1)
        if self.debug:
            self.log(f"Reading image {self.sequence_dir[true_idx]}")
        self.frame_idx += 1
        name = os.path.basename(self.sequence_dir[true_idx])
        return name, np.ascontiguousarray(cv2.imread(self.sequence_dir[true_idx])[..., [2,1,0]]) # BGR to RGB


def test(path, seq_name):
    import matplotlib.pyplot as plt
    d = VideoFrameDataset(path, seq_name)
    for i in range(75):
        img = d[0]
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    test("/home/guanxiux/project/ParallelInference/data/DAVIS/JPEGImages/1080p", "tennis")