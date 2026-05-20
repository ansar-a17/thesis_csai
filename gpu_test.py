import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    x = torch.rand(5, 3).cuda()
    print(f"\nTest tensor on GPU:\n{x}")