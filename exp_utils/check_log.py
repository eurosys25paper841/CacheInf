import os
from os import path as osp
work_dir = os.environ["work"]

path = osp.join(work_dir, "log")
num = 3

print(f"Items that have problem:")
for item in sorted(os.listdir(path)):
    _path = osp.join(path, item)
    if len(os.listdir(_path)) != num:
        print(_path)
        continue
    logs = os.listdir(_path)
    logs.remove("power_record.txt")
    for log in logs:
        with open(osp.join(_path, log)) as f:
            lines = f.readlines()
            success = False
            for l in lines:
                if " 300 th " in l:
                    success = True
                    break
            if not success:
                print(osp.join(_path, log))
                break
    