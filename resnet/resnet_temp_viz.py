from __future__ import annotations

import torch
from pathlib import Path
"""
Please look at `resnet_train.py` for the training script.
"""
import tqdm
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Directory where gradient norms are saved
grad_norms_dir = Path('results') /'gradients' / '2024-08-21-22-01-06' # '2024-08-21-20-05-12' # NOTE this changes sometimes

def exponential_moving_average(data, alpha):
    """
    Compute exponential moving average of data.
    
    :param data: List of values to smooth
    :param alpha: Smoothing factor (0 < alpha <= 1)
    :return: List of smoothed values
    """
    ema = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

def visualize_grad_norms(grad_norms_dir: Path, ems_alphas: list[float]):
    parent_dir = Path('results') / 'gradient_visualizations' / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for ema_alpha in ems_alphas:
        print("*"*100)
        print(f"ema_alpha: {ema_alpha}")
        # Directory to save visualizations
        vis_dir = parent_dir / f"ema_{ema_alpha}"
        os.makedirs(vis_dir, exist_ok=True)

        # Dictionary to store gradient norms for each parameter
        param_grad_norms = defaultdict(list)

        def grad_norm_fname(fname: str) -> tuple[int, int]:
            # key: i.e. gradients_norms_0_6.pt => (0, 6)
            try:
                _, b, c = fname.rsplit('_', 2)
            except Exception as e:
                print(fname)
                raise e
            x = int(b)
            y = int(c.split('.')[0])
            return x, y

        fs = os.listdir(grad_norms_dir)
        _ = len(fs)
        fs = [f for f in fs if "epoch_size" not in f]
        filename_tuples = [grad_norm_fname(x) for x in fs]
        # This is all sanity shit that isn't really too necessary
        # mn = min(max[x for _, x in filename_tuples])
        # mx = max([x for _, x in filename_tuples])
        # mx = 390 # Kind of a hack :P
        # count = max([x for x, _ in filename_tuples]) + 1
        # # assert mn == mx, f"Min: {mn}, Max: {mx}"
        # assert count >= 1
        # assert len(filename_tuples) == (mx+1) * count, f"Expected {mx * count} filenames, got {len(filename_tuples)}: {sorted(set(filename_tuples) - set([(y, x) for x in range(mx) for y in range(count)]))}"
        # print(filename_tuples)
        # Collect gradient norms
        
        assert len(fs) == _ - 1
        for filename in tqdm.tqdm(sorted(fs, key=grad_norm_fname)):
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
            norms_array = np.array(norms)
            assert None not in norms_array
            ema_norms = exponential_moving_average(norms_array, ema_alpha) if ema_alpha is not None else norms_array
            
            plt.figure(figsize=(10, 6))
            plt.plot(ema_norms)
            plt.title(f'Gradient Norm Over Time: {param_name}, ema_alpha={ema_alpha}')
            plt.xlabel('Iteration')
            plt.ylabel('L2 Norm of Gradient')
            plt.yscale('log')  # Use log scale for y-axis - suggestion from claude :P
            
            # Replace dots and slashes in parameter names with underscores for the filename
            safe_param_name = param_name.replace('.', '_').replace('/', '_')
            plt.savefig(os.path.join(vis_dir, f'{safe_param_name}_grad_norm.png'))
            plt.close()

        print(f"Visualizations saved in {vis_dir}")

if __name__ == "__main__":


    # EMA smoothing factor
    # Around 0.01 - 0.1 is best
    # ema_alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, None] # Bigger = less smoothing
    # ema_alphas = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05] # For smoothing that is better
    ema_alphas = [None, 0.08, 0.15]
    visualize_grad_norms(grad_norms_dir, ema_alphas)