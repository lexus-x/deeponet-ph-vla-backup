import sys, numpy as np
from PIL import Image
OUT="/tmp/claude-1000/-home-user-Desktop-Ayush-PH-test/20b8ae9f-eeb6-4e70-8000-90d6cc07d17f/scratchpad"
mode=sys.argv[1]
def save(t,p):
    a=(t[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    Image.fromarray(a).save(p); return round(a.mean(),1),round(a.std(),1),a.shape

if mode=="base":
    import evaluate_act as E
    from libero_v_wrapper import _make_base_libero_env
    env=_make_base_libero_env(0, suite_name="libero_object")
    obs,_=env.reset(seed=0)
    bi=E.env_obs_to_policy_input(obs, env.task_description)
    print("BASE agent:", save(bi["observation.images.image"], f"{OUT}/base_agent.png"),
          "wrist:", save(bi["observation.images.wrist_image"], f"{OUT}/base_wrist.png"))
    print("task:", env.task_description)
else:
    import evaluate_plus_act as P
    bench,tasks=P.list_perturbed_tasks("libero_object")
    env=P.LiberoPlusEnv(bench, tasks[0]["index"], img_size=256)
    obs=env.reset(seed=0)
    pi=P.plus_obs_to_policy_input(obs, env.task_description)
    print("PLUS agent:", save(pi["observation.images.image"], f"{OUT}/plus_agent.png"),
          "wrist:", save(pi["observation.images.wrist_image"], f"{OUT}/plus_wrist.png"))
    print("task:", env.task_description)
