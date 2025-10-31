#!/usr/bin/env python3
"""
reconstruct_from_manifest.py

Reconstructs a SimpleNet model from shard files and manifest.json.
Verifies output against original TorchScript model.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import OrderedDict

# ---- Mapping of layer_type -> nn.Module ----
_LAYER_MAP = {
    "Conv2d": nn.Conv2d,
    "BatchNorm2d": nn.BatchNorm2d,
    "ReLU": nn.ReLU,
    "MaxPool2d": nn.MaxPool2d,
    "Dropout2d": nn.Dropout2d,
    "Sequential": nn.Sequential,
    "Linear": nn.Linear,
}

# ---- Load shards on demand ----
class ShardLoader:
    def __init__(self, shards_dir):
        self.shards_dir = shards_dir
        self._cache = {}

    def load_shard(self, fname):
        if fname not in self._cache:
            path = os.path.join(self.shards_dir, fname)
            self._cache[fname] = torch.load(path, map_location="cpu")
        return self._cache[fname]

    def get_tensor(self, fname, name):
        shard = self.load_shard(fname)
        return shard[name]

# ---- Rebuild layer from manifest entry ----
def build_layer(layer_info, loader, device):
    layer_type = layer_info["layer_type"]
    config = layer_info["config"]
    layer_cls = _LAYER_MAP.get(layer_type)
    if layer_cls is None:
        raise NotImplementedError(f"Layer type {layer_type} not implemented")
    # handle empty config
    layer = layer_cls(**config) if config else layer_cls()
    # load parameters
    for p in layer_info.get("parameters", []):
        tensor = loader.get_tensor(layer_info["shard_file"], p["name"]).to(device)
        setattr(layer, p["name"], nn.Parameter(tensor))
    # load buffers
    for b in layer_info.get("buffers", []):
        tensor = loader.get_tensor(layer_info["shard_file"], b["name"]).to(device)
        layer.register_buffer(b["name"], tensor)
    return layer

# ---- Rebuild full network ----
def build_model_from_manifest(manifest_path, shards_dir, device):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    layers = []
    loader = ShardLoader(shards_dir)
    for layer_info in manifest["layers"]:
        layer = build_layer(layer_info, loader, device)
        layers.append(layer)
    model = nn.Sequential(*layers)
    model.eval()
    return model

# ---- Compare outputs ----
def compare_tensors(t1, t2, tol=1e-4):
    if t1.shape != t2.shape:
        return False, f"Shape mismatch {t1.shape} vs {t2.shape}"
    diff = (t1 - t2).abs()
    return diff.max().item() <= tol, f"max={diff.max().item()}, mean={diff.mean().item()}"

# ---- CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_model", required=True, help="TorchScript model path")
    parser.add_argument("--manifest", required=True, help="manifest.json path")
    parser.add_argument("--shards_dir", required=True, help="directory containing shard files")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--H", type=int, default=224)
    parser.add_argument("--W", type=int, default=224)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load original model
    orig = torch.jit.load(args.orig_model, map_location=device)
    orig.eval()
    dummy = torch.randn(args.batch, 3, args.H, args.W, device=device)
    with torch.no_grad():
        ref_out = orig(dummy).cpu()
    print("Reference output shape:", ref_out.shape)

    # rebuild from shards + manifest
    model = build_model_from_manifest(args.manifest, args.shards_dir, device)
    dummy_cpu = dummy.cpu()
    with torch.no_grad():
        out = model(dummy_cpu.to(device)).cpu()

    print("Reconstructed output shape:", out.shape)
    ok, msg = compare_tensors(ref_out, out, tol=args.tol)
    if ok:
        print("SUCCESS: outputs match within tolerance:", msg)
    else:
        print("FAILURE: outputs differ:", msg)

if __name__ == "__main__":
    main()
