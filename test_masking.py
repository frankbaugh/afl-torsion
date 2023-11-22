import torch

# Assuming you have a tensor with shape [N, C, H, W, 4, 2]
tensor = torch.randn((5, 3, 10, 10, 4, 2))  # Replace with your actual tensor

# Flatten the last two dimensions
flattened_tensor = tensor.view(tensor.size()[:-2] + (-1,))

# Alternatively, you can use the reshape method
# flattened_tensor = tensor.reshape(tensor.size()[:-2] + (-1,))

# Print the sizes of the original and flattened tensors
print("Original Tensor Shape:", tensor.shape)
print("Flattened Tensor Shape:", flattened_tensor.shape)