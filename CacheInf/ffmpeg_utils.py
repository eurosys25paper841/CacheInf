import torch
import torchaudio
from torchaudio.utils import ffmpeg_utils

def check_availability():
    
    print(torch.__version__)
    print(torchaudio.__version__)
    print("FFmpeg Library versions:")
    for k, ver in ffmpeg_utils.get_versions().items():
        print(f"  {k}:\t{'.'.join(str(v) for v in ver)}")

    print("Available NVENC Encoders:")
    for k in ffmpeg_utils.get_video_encoders().keys():
        if "nvenc" in k:
            print(f" - {k}")

    print("Avaialbe GPU:")
    print(torch.cuda.get_device_properties(0))