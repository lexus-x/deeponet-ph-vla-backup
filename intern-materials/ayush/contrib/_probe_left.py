import sys
print("py", sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("gpu", i, torch.cuda.get_device_name(i))
except Exception as e:
    print("TORCH_FAIL", type(e).__name__, e)

for mod in ("lerobot", "libero", "robosuite", "mujoco"):
    try:
        m = __import__(mod)
        print(mod, "OK", getattr(m, "__file__", "?")[:120])
    except Exception as e:
        print(mod, "FAIL", type(e).__name__, e)
