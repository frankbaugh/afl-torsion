import torch

# RESA, RESB => [2, 4, 2]
tensor = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]
    ])

tensor_avg = torch.sum(tensor, dim=-1)

# Create a specific sample mask of shape [N, 4]. one mask per AA
mask = torch.tensor([
    [1, 1, 0, 0],
    [1, 0, 1, 0]
])

# Use .unsqueeze(-1) to add a new dimension to the mask

# Multiply the tensor and the expanded mask
result = tensor_avg * mask

# Print the result
print(result)
