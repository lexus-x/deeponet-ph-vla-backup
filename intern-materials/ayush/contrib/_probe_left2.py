print("hi")
import sys
print(sys.executable)
try:
    import torch
    print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
except Exception as e:
    print("torch", e)
for mod in ("lerobot", "libero", "robosuite", "mujoco"):
    try:
        __import__(mod)
        print(mod, "OK")
    except Exception as e:
        print(mod, type(e).__name__, str(e)[:100])
