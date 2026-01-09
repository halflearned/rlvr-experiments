"""Test whether torch.compile produces identical outputs for train vs eval mode.

For models without dropout/batchnorm (like Qwen3 with LayerNorm only),
train() vs eval() should produce identical outputs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_compile_train_eval_determinism():
    """Test that compiled model produces same output in train vs eval mode."""
    model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B"

    # Load model twice with same weights
    model1 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # Set different modes
    model1.train()
    model2.eval()

    # Disable gradients for both (we only care about forward pass values)
    model1.requires_grad_(False)
    model2.requires_grad_(False)

    # Create test input
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    print("=== Without compile ===")
    with torch.no_grad():
        out1_no_compile = model1(**inputs).logits
        out2_no_compile = model2(**inputs).logits

    diff_no_compile = (out1_no_compile - out2_no_compile).abs().max().item()
    print(f"Max diff (no compile): {diff_no_compile}")
    assert diff_no_compile == 0.0, f"Expected 0.0, got {diff_no_compile}"
    print("PASS: No compile produces identical outputs\n")

    print("=== With compile ===")
    model1_compiled = torch.compile(model1)
    model2_compiled = torch.compile(model2)

    with torch.no_grad():
        # Warm up (first call triggers compilation)
        _ = model1_compiled(**inputs).logits
        _ = model2_compiled(**inputs).logits

        # Actual test
        out1_compiled = model1_compiled(**inputs).logits
        out2_compiled = model2_compiled(**inputs).logits

    diff_compiled = (out1_compiled - out2_compiled).abs().max().item()
    print(f"Max diff (compiled): {diff_compiled}")

    if diff_compiled == 0.0:
        print("PASS: Compiled models produce identical outputs")
    else:
        print(f"FAIL: Compiled models differ by {diff_compiled}")
        print("\nThis appears to be a torch.compile bug - train/eval mode")
        print("should not affect output for models without dropout/batchnorm.")

        # Additional diagnostics
        print("\n=== Diagnostics ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")

        # Check where differences occur
        diff_mask = (out1_compiled - out2_compiled).abs() > 0
        num_diff = diff_mask.sum().item()
        total = diff_mask.numel()
        print(f"Differing elements: {num_diff}/{total} ({100*num_diff/total:.2f}%)")

    return diff_compiled == 0.0


def test_compile_same_mode_determinism():
    """Test that two separately compiled models in same mode produce identical output."""
    model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B"

    model1 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # Both in eval mode
    model1.eval()
    model2.eval()
    model1.requires_grad_(False)
    model2.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    print("=== Both eval, both compiled ===")
    model1_compiled = torch.compile(model1)
    model2_compiled = torch.compile(model2)

    with torch.no_grad():
        _ = model1_compiled(**inputs).logits
        _ = model2_compiled(**inputs).logits

        out1 = model1_compiled(**inputs).logits
        out2 = model2_compiled(**inputs).logits

    diff = (out1 - out2).abs().max().item()
    print(f"Max diff (both eval, compiled): {diff}")

    if diff == 0.0:
        print("PASS: Same-mode compiled models produce identical outputs")
    else:
        print(f"FAIL: Same-mode compiled models differ by {diff}")

    return diff == 0.0


if __name__ == "__main__":
    print("Testing torch.compile determinism for Qwen3-0.6B\n")
    print("=" * 60)

    test1_pass = test_compile_train_eval_determinism()
    print("\n" + "=" * 60 + "\n")
    test2_pass = test_compile_same_mode_determinism()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  train vs eval compiled: {'PASS' if test1_pass else 'FAIL'}")
    print(f"  same mode compiled:     {'PASS' if test2_pass else 'FAIL'}")
