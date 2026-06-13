import torch
import numpy as np
import matplotlib.pyplot as plt

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
x_data = x_data @ x_data.T

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

tensor = torch.rand(3,4)

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Device: {device}\n")

if torch.accelerator.is_available():
  tensor = tensor.to(torch.accelerator.current_accelerator())

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


shape = (2,3,)
zeros_tensor = torch.zeros(shape)
print(f"\n\nShape of tensor: {zeros_tensor.shape}")
print(f"Datatype of tensor: {zeros_tensor.dtype}")
print(f"Device tensor is stored on: {zeros_tensor.device}")
print(f"Zeros tensor: \n{zeros_tensor}")
