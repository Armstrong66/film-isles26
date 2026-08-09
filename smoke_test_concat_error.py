#!/usr/bin/env python
"""
Smoke test to verify the concat error in model.py
This tests the DecoderBlock's concatenation logic.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_upsample_skip_concat():
    """Test that upsampled x can be concatenated with skip connection."""

    # Simulate the decoder block scenario
    # After upsample, x should match skip spatially

    # Scenario 1: Perfect match (H=128, input to bottleneck is 8x8x8)
    x = torch.randn(2, 256, 16, 16, 16)  # After upsample from 8x8x8
    skip = torch.randn(2, 128, 16, 16, 16)  # From encoder skip

    x_upsampled = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
    concat = torch.cat([x_upsampled, skip], dim=1)
    print(f"Test 1 (perfect match): x={x.shape}, skip={skip.shape}, concat={concat.shape}")
    assert concat.shape == (2, 384, 16, 16, 16), "Test 1 failed"

    # Scenario 2: Mismatch (H=129, input to bottleneck is 8x8x8 but skip is 9x9x9)
    x = torch.randn(2, 256, 16, 16, 16)  # After upsample from 8x8x8
    skip = torch.randn(2, 128, 17, 17, 17)  # From encoder skip (odd input)

    x_upsampled = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
    concat = torch.cat([x_upsampled, skip], dim=1)
    print(f"Test 2 (odd input): x={x.shape}, skip={skip.shape}, concat={concat.shape}")
    assert concat.shape == (2, 384, 17, 17, 17), "Test 2 failed"

    # Scenario 3: Mismatch where skip is smaller (this is the bug case)
    # If the encoder downsample truncates differently than upsample doubles
    x = torch.randn(2, 256, 2, 2, 2)  # After upsample from 1x1x1
    skip = torch.randn(2, 128, 1, 1, 1)  # From encoder skip - BUG: skip is smaller!

    # The current code does: if x.shape[2:] != skip.shape[2:]: x = F.interpolate(x, size=skip.shape[2:])
    # This would make x=(2, 256, 1, 1, 1) and skip=(2, 128, 1, 1, 1)
    # Then concat would be (2, 384, 1, 1, 1)
    x_upsampled = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
    concat = torch.cat([x_upsampled, skip], dim=1)
    print(f"Test 3 (skip smaller): x={x.shape}, skip={skip.shape}, concat={concat.shape}")
    assert concat.shape == (2, 384, 1, 1, 1), "Test 3 failed"

    print("\nAll tests passed!")


def test_decoder_block_forward():
    """Test the actual DecoderBlock forward pass."""
    from pipeline.model import DecoderBlock

    # Test with matching shapes
    db = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
    x = torch.randn(2, 256, 16, 16, 16)
    skip = torch.randn(2, 128, 16, 16, 16)
    output = db(x, skip)
    print(f"DecoderBlock test: input={x.shape}, skip={skip.shape}, output={output.shape}")
    assert output.shape == (2, 128, 16, 16, 16), "DecoderBlock test failed"

    print("DecoderBlock test passed!")


def test_model_forward():
    """Test the full model forward pass with CPU."""
    from pipeline.model import ISLES26Model
    from omegaconf import DictConfig
    import sys

    # Try to import LLMConditioner to check if sentence-transformers is available
    track = "A"  # Default to FiLM for smoke test (LLM requires internet)
    try:
        from pipeline.conditioning import LLMConditioner
        # Check if sentence-transformers can actually load (not just import)
        # by checking if we have internet connectivity
        import socket
        socket.setdefaulttimeout(2)
        try:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
            track = "C"
            print("Using Track C (LLM conditioning)")
        except (socket.timeout, OSError):
            print("Using Track A (FiLM conditioning) - internet not available for LLM")
    except ImportError:
        print("Using Track A (FiLM conditioning) - LLM not installed")

    # Create a minimal config
    cfg = DictConfig({
        "model": {
            "size": "small"
        },
        "conditioning": {
            "track": track,
            "film": {
                "metadata_dim": 5,
                "hidden_dim": 64
            },
            "llm": {
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "hidden_dim": 128,
                "freeze_llm": True
            }
        }
    })

    model = ISLES26Model(cfg)

    # Test with small input
    image = torch.randn(2, 1, 64, 64, 64)  # Smaller for faster test
    # Track A expects 5-dim meta_vec, Track C expects 4-dim (as per dataset)
    meta_dim = 5 if track == "A" else 4
    meta_vec = torch.randn(2, meta_dim)
    meta_text = ["test", "test2"]

    # This should not crash with the concat error
    try:
        outputs = model(image, meta_vec, meta_text)
        print(f"Model forward test: input={image.shape}, outputs={[o.shape for o in outputs]}")
        print("Model forward test passed!")
    except RuntimeError as e:
        if "Sizes of tensors must match" in str(e):
            print(f"CONCAT ERROR DETECTED: {e}")
            return False
        raise

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Running smoke tests for concat error...")
    print("=" * 60)

    test_upsample_skip_concat()
    print()

    try:
        test_decoder_block_forward()
    except Exception as e:
        print(f"DecoderBlock test failed: {e}")

    print()
    try:
        test_model_forward()
    except Exception as e:
        print(f"Model forward test failed: {e}")

    print()
    print("All basic tests completed.")
