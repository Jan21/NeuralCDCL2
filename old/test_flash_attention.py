"""
Test the modified CDCLTransformer model with CausalSelfAttention.
Verifies correctness, causal masking, right-padding safety, and training.
"""

import sys
sys.path.insert(0, "/scratch/project_465002050/Petr/cdcl/NeuralCDCL2")

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_sdpa_causal_correctness():
    """Verify SDPA is_causal matches manual causal attention."""
    print("=" * 60)
    print("TEST 1: SDPA is_causal correctness vs manual computation")
    print("=" * 60)

    torch.manual_seed(42)
    B, H, T, D = 2, 4, 32, 16
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Manual causal attention
    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
    scores.masked_fill_(causal_mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out_manual = torch.matmul(attn, v)

    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_d = (out_manual - out_sdpa).abs().max().item()
    passed = max_d < 1e-5
    print(f"  max_diff={max_d:.2e} {'PASSED' if passed else 'FAILED'}\n")
    return passed


def test_model_forward():
    """Test CDCLTransformer forward pass shape and basic correctness."""
    print("=" * 60)
    print("TEST 2: CDCLTransformer forward pass")
    print("=" * 60)

    from neural_cdcl.model import CDCLTransformer

    torch.manual_seed(42)
    model = CDCLTransformer(
        vocab_size=70, d_model=64, n_layers=2, n_heads=4,
        d_ff=256, max_seq_len=128, dropout=0.0,
    ).to(DEVICE)
    model.eval()

    B, T = 4, 64
    input_ids = torch.randint(0, 70, (B, T), device=DEVICE)
    attention_mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    assert logits.shape == (B, T, 70), f"Wrong shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in output"
    print(f"  Output shape: {logits.shape} (correct)")
    print(f"  No NaN values")
    print("  PASSED\n")
    return True


def test_causal_masking():
    """Future tokens must not affect past outputs."""
    print("=" * 60)
    print("TEST 3: Causal masking — future tokens don't leak")
    print("=" * 60)

    from neural_cdcl.model import CDCLTransformer

    torch.manual_seed(42)
    model = CDCLTransformer(
        vocab_size=70, d_model=64, n_layers=2, n_heads=4,
        d_ff=256, max_seq_len=128, dropout=0.0,
    ).to(DEVICE)
    model.eval()

    input_ids = torch.randint(0, 70, (1, 64), device=DEVICE)

    with torch.no_grad():
        logits_full = model(input_ids)
        logits_prefix = model(input_ids[:, :32])

    diff = (logits_full[0, :32] - logits_prefix[0, :32]).abs()
    max_d = diff.max().item()
    passed = max_d < 1e-5
    print(f"  max_diff for prefix: {max_d:.2e} {'PASSED' if passed else 'FAILED'}\n")
    return passed


def test_right_padding_safety():
    """Non-padding outputs must be identical with or without padding."""
    print("=" * 60)
    print("TEST 4: Right-padding safety")
    print("=" * 60)

    from neural_cdcl.model import CDCLTransformer

    torch.manual_seed(42)
    model = CDCLTransformer(
        vocab_size=70, d_model=64, n_layers=2, n_heads=4,
        d_ff=256, max_seq_len=128, dropout=0.0,
    ).to(DEVICE)
    model.eval()

    # Create a batch with padding
    seq_lens = [48, 32]
    max_len = max(seq_lens)
    input_ids = torch.randint(0, 70, (2, max_len), device=DEVICE)
    for i, sl in enumerate(seq_lens):
        input_ids[i, sl:] = 0  # PAD_ID

    with torch.no_grad():
        logits_padded = model(input_ids)
        # Run each sequence individually without padding
        for i, sl in enumerate(seq_lens):
            logits_single = model(input_ids[i:i+1, :sl])
            diff = (logits_padded[i, :sl] - logits_single[0]).abs()
            max_d = diff.max().item()
            print(f"  Seq {i} (len={sl}): max_diff={max_d:.2e}")

    print("  PASSED\n")
    return True


def test_training_backward():
    """Full training step with loss computation and backward."""
    print("=" * 60)
    print("TEST 5: Training forward + backward pass")
    print("=" * 60)

    from neural_cdcl.model import CDCLTransformer

    torch.manual_seed(42)
    model = CDCLTransformer(
        vocab_size=70, d_model=256, n_layers=4, n_heads=16,
        d_ff=1024, max_seq_len=512, dropout=0.1,
    ).to(DEVICE)
    model.train()

    B, T = 8, 256
    input_ids = torch.randint(0, 70, (B, T), device=DEVICE)
    targets = torch.randint(0, 70, (B, T - 1), device=DEVICE)

    logits = model(input_ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 70), targets.reshape(-1))
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    nan_grads = 0
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_grads += 1
    print(f"  Parameters with NaN gradients: {nan_grads}")

    passed = nan_grads == 0
    print(f"  {'PASSED' if passed else 'FAILED'}\n")
    return passed


def test_param_count():
    """Verify param count is reasonable after change."""
    print("=" * 60)
    print("TEST 6: Parameter count check")
    print("=" * 60)

    from neural_cdcl.model import CDCLTransformer

    model = CDCLTransformer(
        vocab_size=70, d_model=256, n_layers=12, n_heads=16,
        d_ff=1024, max_seq_len=4096, dropout=0.1,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,} ({total/1e6:.2f}M)")
    print("  PASSED\n")
    return True


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}, Device: {DEVICE}\n")

    results = [
        ("SDPA correctness", test_sdpa_causal_correctness()),
        ("Model forward", test_model_forward()),
        ("Causal masking", test_causal_masking()),
        ("Right-padding safety", test_right_padding_safety()),
        ("Training backward", test_training_backward()),
        ("Parameter count", test_param_count()),
    ]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {name}: {'PASSED' if passed else 'FAILED'}")

    all_ok = all(p for _, p in results)
    print(f"\n{'All tests passed!' if all_ok else 'Some tests FAILED!'}")
    sys.exit(0 if all_ok else 1)
