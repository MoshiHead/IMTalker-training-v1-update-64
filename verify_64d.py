"""
verify_64d.py
Run this BEFORE starting renderer training to confirm all 64D changes are correct.

Usage (from repo root):
    python verify_64d.py

All checks must print OK. If anything prints FAIL, re-apply that patch.
"""

import sys
import torch

sys.path.insert(0, ".")

# -----------------------------------------------------------------------
# Minimal args stub — matches what renderer/train.py passes to IMTRenderer
# -----------------------------------------------------------------------
class Args:
    dim_motion = 64
    swin_res_threshold = 128
    num_heads = 8
    window_size = 8
    drop_path = 0.1
    low_res_depth = 2
    depth = 2

args = Args()

# -----------------------------------------------------------------------
# 1. Import and build IMTRenderer
# -----------------------------------------------------------------------
print("Building IMTRenderer ...", end=" ")
try:
    from renderer.models import IMTRenderer
    renderer = IMTRenderer(args)
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# 2. Check MotionEncoder output layer shape
# -----------------------------------------------------------------------
enc_w = renderer.latent_token_encoder.final_linear.weight.shape
print(f"MotionEncoder final_linear.weight shape: {enc_w}", end="  ")
if enc_w == torch.Size([64, 512]):
    print("OK")
else:
    print(f"FAIL — expected (64, 512), got {enc_w}")

# -----------------------------------------------------------------------
# 3. Check MotionDecoder first modulation layer shape
# -----------------------------------------------------------------------
dec_mod = renderer.latent_token_decoder.style_conv_layers[0].conv.modulation.weight.shape
print(f"MotionDecoder modulation[0].weight shape: {dec_mod}", end="  ")
# First StyledConv: in_channel=const_dim=32, style_dim=latent_dim=64 → EqualLinear(64, 32)
if dec_mod == torch.Size([32, 64]):
    print("OK")
else:
    print(f"FAIL — expected (32, 64), got {dec_mod}")

# -----------------------------------------------------------------------
# 4. Check IdentidyAdaptive layer shapes
# -----------------------------------------------------------------------
adapt_in = renderer.adapt.in_layer.weight.shape
adapt_out = renderer.adapt.final_linear.weight.shape
print(f"IdentidyAdaptive in_layer.weight:    {adapt_in}", end="  ")
if adapt_in == torch.Size([512, 576]):   # 512 + 64 = 576
    print("OK")
else:
    print(f"FAIL — expected (512, 576), got {adapt_in}")

print(f"IdentidyAdaptive final_linear.weight: {adapt_out}", end="  ")
if adapt_out == torch.Size([64, 512]):
    print("OK")
else:
    print(f"FAIL — expected (64, 512), got {adapt_out}")

# -----------------------------------------------------------------------
# 5. Run a full forward pass with dummy tensors
# -----------------------------------------------------------------------
print("Running forward pass (B=2, 3x256x256) ...", end=" ")
try:
    renderer.eval()
    with torch.no_grad():
        x_src  = torch.randn(2, 3, 256, 256)
        x_drv  = torch.randn(2, 3, 256, 256)
        out_frame, t_c = renderer(x_drv, x_src)
    print("OK")
    print(f"  Output frame shape:  {out_frame.shape}")
    print(f"  Motion latent shape: {t_c.shape}", end="  ")
    if t_c.shape == torch.Size([2, 64]):
        print("OK — 64D confirmed!")
    else:
        print(f"FAIL — expected (2, 64), got {t_c.shape}")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()

# -----------------------------------------------------------------------
# 6. Check mot_encode output directly
# -----------------------------------------------------------------------
print("Checking mot_encode output ...", end=" ")
try:
    with torch.no_grad():
        z = renderer.mot_encode(torch.randn(1, 3, 256, 256))
    print(f"shape={z.shape}", end="  ")
    if z.shape == torch.Size([1, 64]):
        print("OK")
    else:
        print(f"FAIL — expected (1, 64)")
except Exception as e:
    print(f"FAIL: {e}")

print("\n--- All checks complete ---")
