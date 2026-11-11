"""Check that PyTorch sees your GPU and that the model fits in its memory.

    python check_gpu.py
"""
import torch

from grai.model import MultiTaskLoss, MultiTaskModel

print("PyTorch", torch.__version__, "| CUDA build:", torch.version.cuda)
if not torch.cuda.is_available():
    print("\nNo GPU visible to PyTorch. Install the CUDA build of PyTorch, e.g.:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print("and make sure the NVIDIA driver is installed (run `nvidia-smi`).")
    raise SystemExit(1)

dev = "cuda"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | {p.total_memory / 1024**3:.1f} GB")

# one training step with the paper's batch: 2 images (256x256) + ~8 tooth crops (224x224)
model = MultiTaskModel(encoder_weights=None).to(dev).train()
loss_fn = MultiTaskLoss().to(dev)
opt = torch.optim.AdamW(model.parameters())
scaler = torch.amp.GradScaler("cuda")
img = torch.randn(2, 3, 256, 256, device=dev)
mask = (torch.rand(2, 1, 256, 256, device=dev) > 0.5).float()
crops = torch.randn(8, 3, 224, 224, device=dev)
labels = torch.randint(0, 3, (8,), device=dev)
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.float16):
    s, c = model(img, crops)
loss, _, _ = loss_fn(s.float(), mask, c.float(), labels)
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"Training step OK - peak GPU memory {peak:.2f} GB (mixed precision)")
if peak > 0.85 * p.total_memory / 1024**3:
    print("Memory is tight: lower data.crop_size (e.g. 160) in config.yaml.")
