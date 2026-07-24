"""
generate_answer_key.py
-----------------------
Downloads TinyLlama, extracts layer-0's real weights, and computes the
correct (ground-truth) outputs for every function the team is building:

    - rmsnorm      (Hanan)
    - silu, mul    (Kiren)
    - linear_nobias (Jazim)
    - rope         (Shajia)

All inputs, weights, and correct outputs are saved to .npy files in
team-c/test/results/ so every teammate can load them and compare their
C kernel's output against real TinyLlama numbers (not random data).
"""

import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "test/results"
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog and runs into the forest."

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name, array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = array.astype(np.float32)
    path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(path, array)
    print(f"Saved {name}: shape={array.shape} -> {path}")


def main():
    print(f"Loading {MODEL_NAME} (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    config = model.config
    eps = config.rms_norm_eps
    print(f"rms_norm_eps from model config: {eps}")
    save("rmsnorm_eps", np.array([eps], dtype=np.float32))

    inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt")
    with torch.no_grad():
        embed_out = model.model.embed_tokens(inputs["input_ids"])
    hidden_states = embed_out[0]
    save("layer0_input_hidden_states", hidden_states)

    layer0 = model.model.layers[0]

    rmsnorm_module = layer0.input_layernorm
    rmsnorm_weight = rmsnorm_module.weight
    save("rmsnorm_weight", rmsnorm_weight)

    with torch.no_grad():
        rmsnorm_output = rmsnorm_module(hidden_states)
    save("rmsnorm_output", rmsnorm_output)

    q_proj = layer0.self_attn.q_proj
    assert q_proj.bias is None, "Expected q_proj to have no bias!"
    q_weight = q_proj.weight
    save("linear_nobias_weight_qproj", q_weight)

    with torch.no_grad():
        linear_nobias_output = q_proj(rmsnorm_output)
    save("linear_nobias_input", rmsnorm_output)
    save("linear_nobias_output_qproj", linear_nobias_output)

    for proj_name in ["k_proj", "v_proj", "o_proj"]:
        proj = getattr(layer0.self_attn, proj_name)
        assert proj.bias is None, f"Expected {proj_name} to have no bias!"
        save(f"linear_nobias_weight_{proj_name}", proj.weight)

    mlp = layer0.mlp
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        proj = getattr(mlp, proj_name)
        assert proj.bias is None, f"Expected {proj_name} to have no bias!"
        save(f"linear_nobias_weight_{proj_name}", proj.weight)

    with torch.no_grad():
        gate_out = mlp.gate_proj(rmsnorm_output)
        silu_output = torch.nn.functional.silu(gate_out)
    save("silu_input", gate_out)
    save("silu_output", silu_output)

    with torch.no_grad():
        up_out = mlp.up_proj(rmsnorm_output)
        mul_output = silu_output * up_out
    save("mul_input_a", silu_output)
    save("mul_input_b", up_out)
    save("mul_output", mul_output)

    with torch.no_grad():
        k_out = layer0.self_attn.k_proj(rmsnorm_output)
        seq_len = hidden_states.shape[0]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        rotary_emb = model.model.rotary_emb
        cos, sin = rotary_emb(linear_nobias_output.unsqueeze(0), position_ids)

    save("rope_cos", cos[0])
    save("rope_sin", sin[0])
    save("rope_input_q", linear_nobias_output)
    save("rope_input_k", k_out)

    print("\nAll answer-key files saved successfully to:", OUTPUT_DIR)
 


if __name__ == "__main__":
    main()