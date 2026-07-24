# DeepONet overnight SUMMARY

**Updated:** 2026-07-24T15:12:51+09:00

**Verdict:** Plus partial/complete — libero_10: m3=0.000 flow=0.000 d=+0.000; libero_object: m3=0.000 flow=0.000 d=+0.000; libero_spatial: m3=0.446 flow=0.321 d=+0.125

## LIBERO-Plus

### `flow_libero_10`
- **flow** robustness_avg=`0.0` tasks_done=`56`
  - Camera Viewpoints: 0.0
  - Light Conditions: 0.0
  - Sensor Noise: 0.0
  - Background Textures: 0.0
  - Objects Layout: 0.0
  - Robot Initial States: 0.0
  - Language Instructions: 0.0

### `flow_libero_object`
- **flow** robustness_avg=`0.0` tasks_done=`56`
  - Camera Viewpoints: 0.0
  - Light Conditions: 0.0
  - Sensor Noise: 0.0
  - Background Textures: 0.0
  - Objects Layout: 0.0
  - Robot Initial States: 0.0
  - Language Instructions: 0.0

### `flow_libero_spatial`
- **flow** robustness_avg=`0.32142857142857145` tasks_done=`56`
  - Camera Viewpoints: 0.25
  - Light Conditions: 0.5
  - Sensor Noise: 0.375
  - Background Textures: 0.375
  - Objects Layout: 0.5
  - Robot Initial States: 0.25
  - Language Instructions: 0.0

### `m3_libero_10`
- **m3** robustness_avg=`0.0` tasks_done=`56`
  - Camera Viewpoints: 0.0
  - Light Conditions: 0.0
  - Sensor Noise: 0.0
  - Background Textures: 0.0
  - Objects Layout: 0.0
  - Robot Initial States: 0.0
  - Language Instructions: 0.0

### `m3_libero_object`
- **m3** robustness_avg=`0.0` tasks_done=`56`
  - Camera Viewpoints: 0.0
  - Light Conditions: 0.0
  - Sensor Noise: 0.0
  - Background Textures: 0.0
  - Objects Layout: 0.0
  - Robot Initial States: 0.0
  - Language Instructions: 0.0

### `m3_libero_spatial`
- **m3** robustness_avg=`0.44642857142857145` tasks_done=`56`
  - Camera Viewpoints: 0.5
  - Light Conditions: 0.625
  - Sensor Noise: 0.25
  - Background Textures: 0.375
  - Objects Layout: 0.625
  - Robot Initial States: 0.125
  - Language Instructions: 0.625

## POD-30K train

- status: `DONE`
- last_step: `29999`
- latest_ckpt: `30000`

```
[stage2] step  29599 | mse=0.3403 L1=0.9230 PH=0.0000 total=0.3403 | VRAM=50.3GB | 0.638s/it
[stage2] step  29799 | mse=0.3432 L1=0.9236 PH=0.0000 total=0.3432 | VRAM=50.3GB | 0.638s/it
[stage2] step  29999 | mse=0.3374 L1=0.9212 PH=0.0000 total=0.3374 | VRAM=50.3GB | 0.637s/it
[ckpt] saved -> /home/user/Desktop/Ayush PH test/DeepONet PH/v2/pod_train_spatial_30k/checkpoints/30000  (EMA weights)

[train] DONE head=deeponet variant=baseline steps=30000 wall=522.0 min -> /home/user/Desktop/Ayush PH test/DeepONet PH/v2/pod_train_spatial_30k
```

## Long boundary / exec (offline)

- Long ens mean=55.5% vs m3_r5=60.0%; pin status=partial_aborted_for_pod_cpu. Averaging/pinning across mode jumps hurts closed-loop SR (BID-consistent).
- ens: `{"mean_sr": 55.5, "n_tasks": 10, "per_task": {"0": 30.0, "1": 45.0, "2": 90.0, "3": 85.0, "4": 50.0, "5": 85.0, "6": 45.0, "7": 45.0, "8": 5.0, "9": 75.0}}`
- pin: `{"mean_sr": 5.0, "n_tasks": 3, "per_task": {"0": 0.0, "5": 5.0, "6": 10.0}, "status": "partial_aborted_for_pod_cpu", "note": "t0=0% t5=5% t6=10% vs m3_r5=60 \u2014 catastrophic; aborted"}`

## Wake-up checklist

1. If Object/Long Plus m3 >> flow → draft co-lead Plus table
2. Else use POD mid-ckpt + mechanism narrative
3. a100 was left alone (bridge pretrain)
