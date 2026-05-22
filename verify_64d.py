"""
verify_64d.py
Run this BEFORE starting renderer training to confirm all 64D changes are correct.

Usage (from repo root, inside IMTalker conda env):
    python verify_64d.py

All checks must print OK. If anything prints FAIL, re-apply that patch.

NOTE: The model requires 512x512 input images. A smaller size causes a spatial
mismatch inside the cross-attention and will crash — unrelated to 64D.
"""

import sys
import torch

sys.path.insert(0, ".")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

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
print("\n[1] Building IMTRenderer ...", end=" ")
try:
    from renderer.models import IMTRenderer
    renderer = IMTRenderer(args).to(device)
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# 2. Check MotionEncoder output layer shape  (must be [64, 512])
# -----------------------------------------------------------------------
enc_w = renderer.latent_token_encoder.final_linear.weight.shape
print(f"[2] MotionEncoder final_linear shape:       {enc_w}", end="  ")
if enc_w == torch.Size([64, 512]):
    print("OK")
else:
    print(f"FAIL — expected torch.Size([64, 512])")

# -----------------------------------------------------------------------
# 3. Check MotionDecoder first modulation shape  (must be [32, 64])
#    First StyledConv: in_channel=const_dim=32, style_dim=latent_dim=64
# -----------------------------------------------------------------------
dec_mod = renderer.latent_token_decoder.style_conv_layers[0].conv.modulation.weight.shape
print(f"[3] MotionDecoder modulation[0] shape:      {dec_mod}", end="  ")
if dec_mod == torch.Size([32, 64]):
    print("OK")
else:
    print(f"FAIL — expected torch.Size([32, 64])")

# -----------------------------------------------------------------------
# 4. Check IdentidyAdaptive layer shapes
#    in_layer:    dim_app + dim_mot = 512 + 64 = 576 → 512
#    final_linear: 512 → dim_mot = 64
# -----------------------------------------------------------------------
adapt_in  = renderer.adapt.in_layer.weight.shape
adapt_out = renderer.adapt.final_linear.weight.shape
print(f"[4] IdentidyAdaptive in_layer shape:        {adapt_in}", end="  ")
if adapt_in == torch.Size([512, 576]):
    print("OK")
else:
    print(f"FAIL — expected torch.Size([512, 576])")

print(f"[4] IdentidyAdaptive final_linear shape:    {adapt_out}", end="  ")
if adapt_out == torch.Size([64, 512]):
    print("OK")
else:
    print(f"FAIL — expected torch.Size([64, 512])")

# -----------------------------------------------------------------------
# 5. Check mot_encode output dim with a small image (dimension-agnostic)
# -----------------------------------------------------------------------
print("[5] mot_encode(1x3x64x64) output shape ...", end=" ")
try:
    renderer.eval()
    with torch.no_grad():
        z = renderer.mot_encode(torch.randn(1, 3, 64, 64, device=device))
    print(f"{z.shape}", end="  ")
    if z.shape == torch.Size([1, 64]):
        print("OK — 64D confirmed")
    else:
        print(f"FAIL — expected torch.Size([1, 64])")
except Exception as e:
    print(f"FAIL: {e}")

# -----------------------------------------------------------------------
# 6. Full forward pass — MUST use 512x512 (model is designed for this size)
#
#    Why 512x512 is required:
#      IdentityEncoder downsamples 5x (each DownConvResBlock halves spatial).
#      512 → 256 → 128 → 64 → 32 → 16 → 8  (deepest feature: 8x8)
#      MotionDecoder const is 4x4, upsamples to 8x8 for its first output m1.
#      CrossAttention[0] attends A[0]=m1(8x8) against C[0]=f_r[0](8x8) — match!
#      With 256x256 input, deepest feature is 4x4, mismatches m1(8x8) → crash.
# -----------------------------------------------------------------------
print("[6] Full forward pass (B=1, 512x512) ...", end=" ")
try:
    renderer.eval()
    with torch.no_grad():
        x_src = torch.randn(1, 3, 512, 512, device=device)
        x_drv = torch.randn(1, 3, 512, 512, device=device)
        out_frame, t_c = renderer(x_drv, x_src)
    print("OK")
    print(f"    Output frame shape:  {out_frame.shape}")
    print(f"    Motion latent shape: {t_c.shape}", end="  ")
    if t_c.shape == torch.Size([1, 64]):
        print("OK — 64D end-to-end confirmed!")
    else:
        print(f"FAIL — expected torch.Size([1, 64])")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()

print("\n--- All checks complete ---")
print("If checks 2-5 are OK, the 64D architecture is correct.")
print("Check 6 requires a GPU or significant CPU RAM (~4-6 GB for 512x512).")
