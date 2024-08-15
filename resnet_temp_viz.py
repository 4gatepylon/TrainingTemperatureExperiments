from __future__ import annotations

import torch
from pathlib import Path
"""
Please look at `resnet_train.py` for the training script.
"""
import tqdm
import os
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory where gradient norms are saved
grad_norms_dir = Path('results') /'2024-08-15-21-45-30'

# Directory to save visualizations
vis_dir = Path('results') / 'gradient_visualizations'
os.makedirs(vis_dir, exist_ok=True)

# Dictionary to store gradient norms for each parameter
param_grad_norms = defaultdict(list)

def grad_norm_fname(fname: str) -> tuple[int, int]:
    # key: i.e. gradients_norms_0_6.pt => (0, 6)
    _, b, c = fname.rsplit('_', 2)
    x = int(b)
    y = int(c.split('.')[0])
    return x, y

filename_tuples = [grad_norm_fname(x) for x in os.listdir(grad_norms_dir)]
# mn = min(max[x for _, x in filename_tuples])
mx = max([x for _, x in filename_tuples])
count = max([x for x, _ in filename_tuples]) + 1
# assert mn == mx, f"Min: {mn}, Max: {mx}"
assert count >= 1
assert len(filename_tuples) == (mx+1) * count, f"Expected {mx * count} filenames, got {len(filename_tuples)}: {sorted(set(filename_tuples) - set([(y, x) for x in range(mx) for y in range(count)]))}"
# print(filename_tuples)
# Collect gradient norms
for filename in tqdm.tqdm(sorted(os.listdir(grad_norms_dir), key=grad_norm_fname)):
    if filename.endswith('.pt'):
        file_path = os.path.join(grad_norms_dir, filename)
        grad_norms = torch.load(file_path)
        
        for param_name, norm in grad_norms.items():
            if param_name not in param_grad_norms:
                param_grad_norms[param_name] = []
            param_grad_norms[param_name].append(norm)

_ = list(param_grad_norms.values())[0] # not even an iterator smh
assert _ is not None and isinstance(_, list)
assert all(len(v) == len(_) for v in param_grad_norms.values())
print(f"{len(param_grad_norms)} keys, {len(_)} values")

# Create visualizations
for param_name, norms in tqdm.tqdm(param_grad_norms.items()):
    plt.figure(figsize=(10, 6))
    plt.plot(norms)
    plt.title(f'Gradient Norm Over Time: {param_name}')
    plt.xlabel('Iteration')
    plt.ylabel('L2 Norm of Gradient')
    plt.yscale('log')  # Use log scale for y-axis - suggestion from claude :P
    
    # Replace dots and slashes in parameter names with underscores for the filename
    safe_param_name = param_name.replace('.', '_').replace('/', '_')
    plt.savefig(os.path.join(vis_dir, f'{safe_param_name}_grad_norm.png'))
    plt.close()

print(f"Visualizations saved in {vis_dir}")