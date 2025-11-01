#!/usr/bin/env python3
import os
import json
import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertOnlyMLMHead,
)
torch.backends.cudnn.benchmark = True

# ---------- Helpers ----------
ROOT = "/home/vanisha/Documents/Coding/Snac-Underflow"
LAYER_OUT_DIR = os.path.join(ROOT, "layer_outputs")
ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
ensure_dir(LAYER_OUT_DIR)

def save_tensor(tensor, name):
    path = os.path.join(LAYER_OUT_DIR, f"{name}.pt")
    torch.save(tensor.detach().cpu(), path)
    print(f"   [SAVE] Saved output tensor → {path}")

def log_vram(prefix):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        resv  = torch.cuda.memory_reserved() / (1024 ** 2)
        tot  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"   {prefix} | VRAM: {alloc:.1f}MB used / {resv:.1f}MB reserved / {tot:.1f}MB total")
    else:
        print(f"   {prefix} | VRAM: n/a (cuda not available)")

def load_tensors(shard_file):
    obj = torch.load(shard_file, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Shard {shard_file} did not contain a dict state_dict")
    return obj

# ---------- Layer builder ----------
def build_layer(layer_info, bert_config):
    lt = layer_info.get("layer_type", "")
    # Only support these HF classes explicitly; other layer types become Identity
    if lt == "BertEmbeddings":
        return BertEmbeddings(bert_config)
    if lt == "BertLayer":
        return BertLayer(bert_config)
    if lt == "BertOnlyMLMHead":
        return BertOnlyMLMHead(bert_config)
    # fallback to identity for unknown types (keep your logs)
    print(f"[WARN] Unknown layer type '{lt}', using Identity()")
    return nn.Identity()

# ---------- Wrapper module for safe tracing ----------
class ShardedBertModule(nn.Module):
    """
    Wraps a list of layers (BertEmbeddings, BertLayer, BertOnlyMLMHead, etc.)
    and exposes forward(input_ids, attention_mask) returning the final tensor.
    This ensures we handle tuple returns and kwargs correctly for tracing.
    """
    def __init__(self, layers):
        super().__init__()
        # register layers as ModuleList so trace can see them
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids, attention_mask):
        # embeddings -> (batch, seq, hidden)
        x = None
        for layer in self.layers:
            # HuggingFace classes
            if isinstance(layer, BertEmbeddings):
                x = layer(input_ids)                 # embeddings expect input_ids (and maybe token_type_ids/position_ids)
            elif isinstance(layer, BertLayer):
                # BertLayer signature: forward(hidden_states, attention_mask=None, head_mask=None, output_attentions=False)
                # It returns a tuple: (hidden_states, attn_probs?) in HF impl. We pick [0].
                out = layer(x, attention_mask=attention_mask)
                # many HF layers return tuple (hidden_states, present, etc.) — pick first
                if isinstance(out, tuple):
                    x = out[0]
                else:
                    x = out
            elif isinstance(layer, BertOnlyMLMHead):
                x = layer(x)  # mlm head expects hidden-states
            else:
                # generic torch layers (Identity etc.)
                x = layer(x)
        return x

# ---------- Main ----------
def main():
    print("\n=== SNAC-UNDERFLOW SHARDED MODEL RUNNER (TRACE-SAFE) ===")
    manifest = os.path.join(ROOT, "sharding", "shardones", "manifest.json")
    shard_dir = os.path.join(ROOT, "sharding", "shardones")
    out_path = os.path.join(ROOT, "scripted_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] manifest: {manifest}")
    print(f"[INFO] shard_dir: {shard_dir}\n")

    # load BERT config and patch attention implementation (HF versions)
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    # ensure there's a default value for the internal field used by HF
    if not hasattr(bert_config, "_attn_implementation") or bert_config._attn_implementation is None:
        bert_config._attn_implementation = "eager"

    # get input text once
    text = input("Enter input text (with optional [MASK]): ").strip()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    print(f"[INFO] Tokenized input shape: {tuple(input_ids.shape)}\n")

    # read manifest
    data = json.load(open(manifest, "r"))
    layers_meta = data.get("layers", [])
    built_layers = []

    # build and load each layer
    for i, info in enumerate(layers_meta, 1):
        lid = info.get("layer_id", f"layer{i:03d}")
        shard_file = info.get("shard_file")
        print(f"[BUILD] Layer {i}: {lid}")

        layer = build_layer(info, bert_config)
        if shard_file:
            path = os.path.join(shard_dir, shard_file)
            if not os.path.exists(path):
                print(f"   [ERROR] shard file missing: {path} — skipping load")
            else:
                sd = load_tensors(path)
                missing, unexpected = layer.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    print(f"   [WARN] Missing: {missing or 'None'}, Unexpected: {unexpected or 'None'}")
                print(f"   [OK] Loaded weights from {shard_file}")
        else:
            print("   [WARN] no shard_file specified for this layer; using fresh-initialized layer")
        built_layers.append(layer)
        print()

    # create wrapper module and move to device
    wrapper = ShardedBertModule(built_layers).to(device)
    wrapper.eval()

    # ---- Run forward pass (same code as wrapper.forward) and log per-layer outputs ----
    print("[INFO] Starting forward pass (per-layer) ...\n")
    x = input_ids
    layer_times = []
    with torch.no_grad():
        for idx, layer in enumerate(built_layers, 1):
            start = time.time()
            # same handling as wrapper
            if isinstance(layer, BertEmbeddings):
                x = layer(input_ids)
            elif isinstance(layer, BertLayer):
                out = layer(x, attention_mask=attention_mask)
                x = out[0] if isinstance(out, tuple) else out
            elif isinstance(layer, BertOnlyMLMHead):
                x = layer(x)
            else:
                x = layer(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = (time.time() - start) * 1000
            layer_times.append(elapsed)

            print(f"   [FORWARD] L{idx:<2} {layer.__class__.__name__:<20} -> {tuple(x.shape)} | {elapsed:.2f} ms")
            save_tensor(x, f"layer_{idx:03d}_output")
            log_vram("After layer")
            # simulate eviction if memory too high (customizable)
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                print("   [EVICT] VRAM threshold exceeded — emptying cache (simulated eviction)")
                torch.cuda.empty_cache()

    print("\n[OK] Forward complete.")
    total = sum(layer_times)
    print(f"[METRIC] total inference time: {total:.2f} ms")
    print(f"[METRIC] avg per-layer time: { (total / len(layer_times)) if layer_times else 0:.2f} ms")

    # if MLM head present and we had a mask, decode
# ---- Final output decoding ----
    final_out = x

    if final_out.dim() == 3:
        # If MLM case, predict masked tokens only
        if "[MASK]" in text:
            mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if mask_idx.numel() > 0:
                mask_logits = final_out[0, mask_idx, :]
                topk = torch.topk(mask_logits, k=5, dim=-1)
                print("\n[OUTPUT] Top predictions for [MASK]:")
                for k in range(topk.indices.shape[1]):
                    token_id = topk.indices[0, k].item()
                    score = topk.values[0, k].item()
                    print(f"   {tokenizer.decode([token_id])} ({score:.3f})")

                # Construct sentence with best token
                best_token_id = topk.indices[0, 0].item()
                predicted_token = tokenizer.decode([best_token_id])
                reconstructed = text.replace("[MASK]", predicted_token)
                print(f"\n[TEXT OUTPUT] → {reconstructed}")

        # If not MLM, decode full sequence directly
        else:
            pred_ids = torch.argmax(final_out, dim=-1)
            decoded = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            print(f"\n[TEXT OUTPUT] → {decoded}")


    # ---- Optional masked token decoding ----
        if "[MASK]" in text and final_out.dim() == 3:
            mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if mask_idx.numel() > 0:
                mask_logits = final_out[0, mask_idx, :]
                topk = torch.topk(mask_logits, k=5, dim=-1)
                print("\n[OUTPUT] Top predictions for [MASK]:")
                for k in range(topk.indices.shape[1]):
                    token_id = topk.indices[0, k].item()
                    score = topk.values[0, k].item()
                    print(f"   {tokenizer.decode([token_id])} ({score:.3f})")

                # === Construct full sentence with top prediction ===
                # === Construct full sentences for top-k predictions ===
                print("\n[TEXT OUTPUTS]")
                for k in range(topk.indices.shape[1]):
                    token_id = topk.indices[0, k].item()
                    score = topk.values[0, k].item()
                    predicted_token = tokenizer.decode([token_id])
                    reconstructed = text.replace("[MASK]", predicted_token)
                    print(f"   {k+1}. {reconstructed}  (score={score:.3f})")



    # ---- Trace wrapper safely ----
    print("\n[INFO] Tracing wrapper module to TorchScript — using both input_ids and attention_mask")
    try:
        # ensure wrapper is on CPU or same device as inputs we pass to trace
        trace_inputs = (input_ids, attention_mask)
        scripted = torch.jit.trace(wrapper, trace_inputs, strict=False)
        scripted.save(out_path)
        print(f"[DONE] Scripted model saved to {out_path}")
    except Exception as e:
        print(f"[ERROR] Tracing failed: {e}\n")
        # fallback: try tracing with only input_ids if attention mask causes problems
        try:
            scripted = torch.jit.trace(wrapper, (input_ids,), strict=False)
            scripted.save(out_path)
            print(f"[DONE] Scripted with single input (input_ids) saved to {out_path}")
        except Exception as e2:
            print(f"[ERROR] Fallback trace also failed: {e2}")

    log_vram("[FINAL]")
    print("\nOutputs saved to:", LAYER_OUT_DIR)

if __name__ == "__main__":
    main()
