"""Test whether two separately compiled HF models produce identical outputs.

This tests the hypothesis that separately torch.compile'd models can diverge,
even with identical weights and architecture.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_two_compiled_models():
    """Test that two separately compiled models produce same output."""
    model_path = "/efs/rlvr-experiments/assets/hf/Qwen3-0.6B"

    print("Loading two identical models...")
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

    # Both in same mode
    model1.eval()
    model2.eval()
    model1.requires_grad_(False)
    model2.requires_grad_(False)

    # Create test input
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    print("\n=== Without compile ===")
    with torch.no_grad():
        out1_no_compile = model1(**inputs).logits
        out2_no_compile = model2(**inputs).logits

    diff_no_compile = (out1_no_compile - out2_no_compile).abs().max().item()
    print(f"Max diff (no compile): {diff_no_compile}")

    print("\n=== With compile (separately compiled) ===")
    model1_compiled = torch.compile(model1)
    model2_compiled = torch.compile(model2)

    with torch.no_grad():
        # Warm up
        _ = model1_compiled(**inputs).logits
        _ = model2_compiled(**inputs).logits

        # Actual test
        out1_compiled = model1_compiled(**inputs).logits
        out2_compiled = model2_compiled(**inputs).logits

    diff_compiled = (out1_compiled - out2_compiled).abs().max().item()
    print(f"Max diff (compiled): {diff_compiled}")

    print("\n=== Test different compile backends ===")
    # Test if using same compile config helps
    model3 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model4 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model3.eval()
    model4.eval()

    # Try with deterministic mode
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True

    model3_compiled = torch.compile(model3, mode="reduce-overhead")
    model4_compiled = torch.compile(model4, mode="reduce-overhead")

    with torch.no_grad():
        _ = model3_compiled(**inputs).logits
        _ = model4_compiled(**inputs).logits
        out3 = model3_compiled(**inputs).logits
        out4 = model4_compiled(**inputs).logits

    diff_deterministic = (out3 - out4).abs().max().item()
    print(f"Max diff (deterministic mode): {diff_deterministic}")

    print("\n=== SUMMARY ===")
    print(f"No compile:         {diff_no_compile} {'PASS' if diff_no_compile == 0 else 'FAIL'}")
    print(f"Compiled (default): {diff_compiled} {'PASS' if diff_compiled == 0 else 'FAIL'}")
    print(f"Compiled (determ.): {diff_deterministic} {'PASS' if diff_deterministic == 0 else 'FAIL'}")

    if diff_compiled != 0:
        print("\n*** BUG: Two separately compiled models produce different outputs! ***")
        print("This is unexpected for models without stochastic layers.")


if __name__ == "__main__":
    test_two_compiled_models()
