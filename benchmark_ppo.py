import torch
import time

N = 128
T = 2000
seq_len = 16


def func1(dones, seq_len):
    windows = []
    for n in range(N):
        seg_start = 0
        for t in range(T):
            is_boundary = dones[n, t] > 0.5
            if is_boundary or t == T - 1:
                seg_end = t + 1
                s = seg_start
                while s < seg_end:
                    e = min(s + seq_len, seg_end)
                    if e - s >= 4:
                        windows.append((n, s, e - s))
                    s = e
                seg_start = seg_end
    return windows

def func4(dones, seq_len):
    windows = []
    N, T = dones.shape

    # Using nonzero() instead of where
    # Create boundary mask
    boundaries = dones > 0.5
    boundaries[:, -1] = True # T-1 is always a boundary

    envs, timesteps = boundaries.nonzero(as_tuple=True)

    envs = envs.tolist()
    timesteps = timesteps.tolist()

    current_env = -1
    seg_start = 0

    for n, t in zip(envs, timesteps):
        if n != current_env:
            seg_start = 0
            current_env = n

        seg_end = t + 1
        s = seg_start
        while s < seg_end:
            e = min(s + seq_len, seg_end)
            if e - s >= 4:
                windows.append((n, s, e - s))
            s = e
        seg_start = seg_end

    return windows

if __name__ == "__main__":
    dones = torch.zeros(N, T)
    # Add some random boundaries
    dones[torch.rand(N, T) < 0.05] = 1.0

    start1 = time.time()
    w1 = func1(dones, seq_len)
    time1 = time.time() - start1

    start4 = time.time()
    w4 = func4(dones, seq_len)
    time4 = time.time() - start4

    print(f"func1: {time1:.4f}s, len: {len(w1)}")
    print(f"func4: {time4:.4f}s, len: {len(w4)}")
    print(f"Match: {w1 == w4}")
