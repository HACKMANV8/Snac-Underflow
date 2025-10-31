# shard_full_model_blueprint_full.py
import os
import json
import torch
import hashlib
from collections import OrderedDict

def tensor_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())

def sha1_bytes(tensor):
    return hashlib.sha1(tensor.detach().cpu().numpy().tobytes()).hexdigest()

def shard_from_blueprint(blueprint_path, model_path, out_dir, max_shard_mb=5):
    # Load blueprint
    with open(blueprint_path, "r") as f:
        blueprint = json.load(f)

    # Load model (handle PyTorch or ScriptModule)
    if model_path.endswith(".pt") or model_path.endswith(".pth"):
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    else:
        raise ValueError("Unsupported model format")

    if not isinstance(model, dict):
        try:
            state_dict = model.state_dict()
        except Exception:
            # ScriptModule
            state_dict = {name: p.detach().cpu() for name, p in model.named_parameters()}
    else:
        state_dict = model

    os.makedirs(out_dir, exist_ok=True)
    manifest = {"shards": [], "layers": []}
    shard_idx = 1
    current_shard = OrderedDict()
    current_bytes = 0

    for layer in blueprint["layers"]:
        layer_entry = {
            "layer_id": layer.get("layer_id"),
            "module_scope": layer.get("module_scope"),
            "layer_type": layer.get("layer_type"),
            "params": []
        }

        for param in layer.get("params", []):
            pname = param["param_full_name"]
            tensor = state_dict.get(pname, None)

            # record param metadata even if None (e.g., non-loaded)
            param_meta = {
                "param_full_name": pname,
                "shape": list(param.get("shape", [])),
                "dtype": param.get("dtype"),
                "numel": param.get("numel"),
                "size_bytes": param.get("size_bytes"),
                "sha1": sha1_bytes(tensor) if tensor is not None else None
            }
            layer_entry["params"].append(param_meta)

            if tensor is None:
                continue  # skip None tensors for actual sharding

            tensor_size = tensor_bytes(tensor)
            if current_bytes + tensor_size > max_shard_mb * 1024 * 1024 and current_shard:
                # Save current shard
                fname = f"shard_{shard_idx:03d}.pt"
                torch.save(current_shard, os.path.join(out_dir, fname))
                manifest["shards"].append({
                    "file": fname,
                    "group": " + ".join(current_shard.keys()),
                    "total_bytes": current_bytes,
                    "n_tensors": len(current_shard),
                    "items": [{
                        "param_name": n,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype).split(".")[-1],
                        "bytes": tensor_bytes(t),
                        "sha1": sha1_bytes(t)
                    } for n, t in current_shard.items()]
                })
                shard_idx += 1
                current_shard = OrderedDict()
                current_bytes = 0

            current_shard[pname] = tensor
            current_bytes += tensor_size

        manifest["layers"].append(layer_entry)

    # Save last shard
    if current_shard:
        fname = f"shard_{shard_idx:03d}.pt"
        torch.save(current_shard, os.path.join(out_dir, fname))
        manifest["shards"].append({
            "file": fname,
            "group": " + ".join(current_shard.keys()),
            "total_bytes": current_bytes,
            "n_tensors": len(current_shard),
            "items": [{
                "param_name": n,
                "shape": list(t.shape),
                "dtype": str(t.dtype).split(".")[-1],
                "bytes": tensor_bytes(t),
                "sha1": sha1_bytes(t)
            } for n, t in current_shard.items()]
        })

    # Save manifest including non-param layers
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Sharding complete. {shard_idx} shards written to {out_dir}")
    return manifest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("blueprint", help="Path to blueprint JSON")
    parser.add_argument("model", help="Path to model checkpoint")
    parser.add_argument("out_dir", help="Directory to save shards")
    parser.add_argument("--max-shard-mb", type=int, default=50)
    args = parser.parse_args()

    shard_from_blueprint(args.blueprint, args.model, args.out_dir, args.max_shard_mb)
