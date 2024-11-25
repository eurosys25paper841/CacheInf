import io
import time
from threading import Thread
from queue import Queue

import av

import torch
import torchaudio

import numpy as np
from numpy.lib import recfunctions as rfn

from torchaudio.io import StreamReader, StreamWriter

import struct

from ffmpeg_utils import check_availability

# check_availability()
AVMotionVector_format = ">iBBhhhhQiiH"
SIZE_AVMotionVector = struct.calcsize(AVMotionVector_format)
assert SIZE_AVMotionVector == 32
"""
# From libavutil/motion_vector.h
typedef struct AVMotionVector {
    /**
     * Where the current macroblock comes from; negative value when it comes
     * from the past, positive value when it comes from the future.
     * XXX: set exact relative ref frame reference instead of a +/- 1 "direction".
     */
    int32_t source;
    /**
     * Width and height of the block.
     */
    uint8_t w, h;
    /**
     * Absolute source position. Can be outside the frame area.
     */
    int16_t src_x, src_y;
    /**
     * Absolute destination position. Can be outside the frame area.
     */
    int16_t dst_x, dst_y;
    /**
     * Extra flag information.
     * Currently unused.
     */
    uint64_t flags;
    /**
     * Motion vector
     * src_x = dst_x + motion_x / motion_scale
     * src_y = dst_y + motion_y / motion_scale
     */
    int32_t motion_x, motion_y;
    uint16_t motion_scale;
} AVMotionVector;
"""

class MotionVectorExporter:
    """Encodes cuda tensor into h264 video,
    then decodes the video to extract motion vectors
    """
    def __init__(self, height=1080, width=1920, threaded=False):
        self.pict_config = {
            "height": height,
            "width": width,
            "frame_rate": 30000 / 1001,
            "format": "yuv444p",
        }
        self.buffer = io.BytesIO()
        self.encoder = StreamWriter(self.buffer, format="sdl")
        self.encoder.add_video_stream(
            **self.pict_config, encoder="h264_nvenc",
            encoder_format="yuv444p", hw_accel="cuda:0")
        self._encoder = self.encoder.open()

    def extract_mvs2(self, ref_frame: torch.Tensor, query_frame: torch.Tensor,
                    device="cuda:0")->torch.Tensor:
        frames = torch.cat([ref_frame, query_frame], 0)
        stime = time.time()
        init_time = stime
        # Start new encoding context to encode the two frames
        self._encoder.write_video_chunk(0, frames, 0.)
        encode_time = time.time()
        self.buffer.seek(0)
        print(self.buffer.getbuffer().nbytes)
        video = av.open(self.buffer)
        video.streams.video[0].codec_context.export_mvs = True
        av_frames = [_frame for _frame in video.decode(video=0)]
        decode_time = time.time()
        assert len(av_frames) == 2
        assert av_frames[0].key_frame
        P_frame: av.VideoFrame = av_frames[-1]
        mvs: np.ndarray = P_frame.side_data.get("MOTION_VECTORS").to_ndarray()
        mvs = rfn.structured_to_unstructured(mvs, dtype=np.int32)
        # Origin: src_x = dst_x + motion_x / motion_scale
        # new_motion_x = motion_x / motion_scale * -1
        # Convert to dst_x = src_x + new_motion_x
        mvs[..., [8,9]] = mvs[..., [8,9]] // -mvs[..., [10]]
        ret_mvs = mvs[np.any(mvs[..., [8,9]]!=0, axis=-1)][..., [1,2,3,4,5,6,8,9]]
        ret_mvs = torch.from_numpy(ret_mvs).to(device)
        get_mvs_time = time.time()
        print(f"init time: {init_time - stime:.4f}s; encode time: {encode_time - init_time:.4f}s"
              f"decode time: {decode_time-encode_time:.4f}s get mvs time{get_mvs_time-decode_time:.4f}s")

    def extract_mvs(self, ref_frame: torch.Tensor, query_frame: torch.Tensor,
                    device="cuda:0")->torch.Tensor:
        frames = torch.cat([ref_frame, query_frame], 0)
        stime = time.time()
        # Start new encoding context to encode the two frames
        buffer = io.BytesIO()
        encoder = StreamWriter(buffer, format="mp4")
        encoder.add_video_stream(
            **self.pict_config, encoder="h264_nvenc",
            encoder_format="yuv444p", hw_accel="cuda:0")
        init_time = time.time()
        with encoder.open():
            encoder.write_video_chunk(0, frames)
        encode_time = time.time()
        buffer.seek(0)
        video = av.open(buffer)
        video.streams.video[0].codec_context.export_mvs = True
        av_frames = [_frame for _frame in video.decode(video=0)]
        decode_time = time.time()
        assert len(av_frames) == 2
        assert av_frames[0].key_frame
        P_frame: av.VideoFrame = av_frames[-1]
        mvs: np.ndarray = P_frame.side_data.get("MOTION_VECTORS").to_ndarray()
        mvs = rfn.structured_to_unstructured(mvs, dtype=np.int32)
        # Origin: src_x = dst_x + motion_x / motion_scale
        # new_motion_x = motion_x / motion_scale * -1
        # Convert to dst_x = src_x + new_motion_x
        mvs[..., [8,9]] = mvs[..., [8,9]] // -mvs[..., [10]]
        ret_mvs = mvs[np.any(mvs[..., [8,9]]!=0, axis=-1)][..., [1,2,3,4,5,6,8,9]]
        ret_mvs = torch.from_numpy(ret_mvs).to(device)
        get_mvs_time = time.time()
        print(f"init time: {init_time - stime:.4f}s; encode time: {encode_time - init_time:.4f}s"
              f"decode time: {decode_time-encode_time:.4f}s get mvs time{get_mvs_time-decode_time:.4f}s")

def nvenc_tutorial():
    pict_config = {
            "height": 340,
            "width": 640,
            "frame_rate": 30000 / 1001,
            "format": "yuv444p",
    }
    def get_data(height, width, format="yuv444p", frame_rate=30000 / 1001, duration=4):
        src = f"testsrc2=rate={frame_rate}:size={width}x{height}:duration={duration}"
        s = StreamReader(src=src, format="lavfi")
        s.add_basic_video_stream(-1, format=format)
        s.process_all_packets()
        (video,) = s.pop_chunks()
        return video

    frame_data = get_data(**pict_config)

    buffer = io.BytesIO()
    w = StreamWriter(buffer, format="mp4")
    w.add_video_stream(**pict_config, encoder="h264_nvenc", encoder_format="yuv444p", hw_accel="cuda:0")
    with w.open():
        w.write_video_chunk(0, frame_data.to(torch.device("cuda:0")))
    buffer.seek(0)
    video_cuda = buffer.read()


def test():
    import cv2
    import os
    work_dir = os.environ.get("work")
    img1 = cv2.imread(
        os.path.join(work_dir, "data/DAVIS/JPEGImages/1080p/bear/00000.jpg"))[..., [2,1,0]]
    img2 = cv2.imread(
        os.path.join(work_dir, "data/DAVIS/JPEGImages/1080p/bear/00001.jpg"))[..., [2,1,0]]
    img3 = cv2.imread(
        os.path.join(work_dir, "data/DAVIS/JPEGImages/1080p/bear/00002.jpg"))[..., [2,1,0]]
    mv_extractor = MotionVectorExporter(img1.shape[0], img1.shape[1], threaded=True)
    ref_frame = torch.from_numpy(img1).movedim(-1, 0)[None].to("cuda:0")
    query_frame = torch.from_numpy(img2).movedim(-1, 0)[None].to("cuda:0")
    query_frame2 = torch.from_numpy(img3).movedim(-1, 0)[None].to("cuda:0")
    
    mv_extractor.extract_mvs2(ref_frame, query_frame)
    mv_extractor.extract_mvs2(ref_frame, query_frame2)
    mv_extractor.extract_mvs(ref_frame, query_frame)
    mv_extractor.extract_mvs(ref_frame, query_frame2)
    mv_extractor.extract_mvs(ref_frame, query_frame)
    mv_extractor.extract_mvs(ref_frame, query_frame2)
    mv_extractor.extract_mvs(ref_frame, query_frame)
    mv_extractor.extract_mvs(ref_frame, query_frame2)

def test_0():
    nvenc_tutorial()

if __name__ == "__main__":
    # test_0()
    test()


