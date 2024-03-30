import torch
import ray
# ray.available_resources()
# Check for CUDA availability
if torch.cuda.is_available():
    print("CUDA devices available:")
    for i in range(torch.cuda.device_count()):
        print(f"  {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available on this system.")

# Assign a tensor on CUDA device if available
if torch.cuda.is_available():
    device = torch.device("cuda")          # Default CUDA device
    tensor_on_gpu = torch.tensor([1, 2, 3]).to(device)
    print("Tensor assigned on CUDA device:", tensor_on_gpu)
else:
    print("CUDA is not available, cannot assign tensor on CUDA device.")