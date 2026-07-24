#!/usr/bin/env python
"""Offline-stats wrapper for evaluate_exec.py.

Stubs LeRobotDatasetMetadata so stats come from the checkpoint normalizer
(avoids online dataset version checks). Does NOT force HF hub offline — the
VLM weights may need to load from ~/.cache or download.
"""
import os
import sys

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from safetensors.torch import load_file

_SF = os.environ["OFFLINE_STATS_SF"]


def _stats_from_ckpt(sf_path):
    flat = load_file(sf_path)
    out = {}
    for fk, v in flat.items():
        feat, stat = fk.rsplit(".", 1)
        out.setdefault(feat, {})[stat] = v.cpu().numpy()
    return out


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Ensure v2 campaign dir is importable when launched from contrib/
_V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "DeepONet PH", "v2")
_V2 = os.path.abspath(_V2)
if os.path.isdir(_V2):
    sys.path.insert(0, _V2)

import evaluate as ev  # noqa: E402


class _StubMeta:
    def __init__(self, *a, **k):
        pass

    @property
    def stats(self):
        return _stats_from_ckpt(_SF)


ev.LeRobotDatasetMetadata = _StubMeta

import evaluate_exec as ee  # noqa: E402

if __name__ == "__main__":
    ee.main()
