import os
import matplotlib.pyplot as plt

save_dir = "learnable_b_results"
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# Data
# -------------------------
epochs = list(range(25, 52))

# Fixed β / regular run
fixed_train = [
    1.9641, 1.9213, 1.9337, 1.8913, 1.8682, 1.8107, 1.8449,
    1.8215, 1.8246, 1.7971, 1.7798, 1.7787, 1.7721, 1.7249,
    1.7081, 1.7304, 1.7127, 1.7130, 1.6947, 1.6680, 1.6732,
    1.6644, 1.6712, 1.6701, 1.6194, 1.6272, 1.6021
]

fixed_val = [
    1.8774, 1.8676, 1.8443, 1.8261, 1.7892, 1.8048, 1.7499,
    1.7535, 1.7143, 1.7107, 1.6955, 1.6677, 1.6690, 1.6671,
    1.6480, 1.6128, 1.6165, 1.5839, 1.5820, 1.5560, 1.5819,
    1.5614, 1.5694, 1.5565, 1.5321, 1.5389, 1.5186
]

# Learnable β run
learn_train = [
    2.0444, 1.9338, 1.9350, 1.8868, 1.8637, 1.8038, 1.8370,
    1.8116, 1.8154, 1.7872, 1.7684, 1.7652, 1.7590, 1.7100,
    1.6911, 1.7152, 1.6975, 1.6963, 1.6803, 1.6501, 1.6545,
    1.6484, 1.6521, 1.6503, 1.5996, 1.6052, 1.5816
]

learn_val = [
    1.9103, 1.8690, 1.8503, 1.8005, 1.7765, 1.8065, 1.7486,
    1.7390, 1.7071, 1.6984, 1.6952, 1.6597, 1.6463, 1.6522,
    1.6289, 1.6079, 1.6097, 1.5702, 1.5585, 1.5261, 1.5755,
    1.5503, 1.5471, 1.5366, 1.5256, 1.5256, 1.4956
]

# Learnable β stats
beta_mean = [
    0.499954, 0.499986, 0.500028, 0.500082, 0.500137, 0.500195,
    0.500260, 0.500323, 0.500360, 0.500405, 0.500470, 0.500526,
    0.500583, 0.500632, 0.500687, 0.500725, 0.500758, 0.500797,
    0.500834, 0.500883, 0.500925, 0.500973, 0.501025, 0.501068,
    0.501101, 0.501134, 0.501176
]

rwkv_mean = [
    0.499948, 0.500035, 0.500146, 0.500250, 0.500357, 0.500472,
    0.500582, 0.500665, 0.500765, 0.500882, 0.501000, 0.501091,
    0.501176, 0.501264, 0.501360, 0.501442, 0.501523, 0.501594,
    0.501672, 0.501758, 0.501835, 0.501907, 0.501994, 0.502080,
    0.502151, 0.502223, 0.502303
]

rffn_mean = [
    0.499959, 0.499938, 0.499909, 0.499914, 0.499918, 0.499918,
    0.499937, 0.499981, 0.499956, 0.499929, 0.499939, 0.499962,
    0.499990, 0.500001, 0.500014, 0.500009, 0.499993, 0.500000,
    0.499995, 0.500007, 0.500014, 0.500038, 0.500057, 0.500056,
    0.500051, 0.500046, 0.500049
]

beta_std = [
    0.000527, 0.000766, 0.000987, 0.001196, 0.001407, 0.001636,
    0.001831, 0.002025, 0.002221, 0.002407, 0.002591, 0.002761,
    0.002934, 0.003102, 0.003266, 0.003429, 0.003588, 0.003751,
    0.003911, 0.004066, 0.004232, 0.004378, 0.004525, 0.004681,
    0.004836, 0.004994, 0.005142
]

beta_max_dev = [
    0.002496, 0.004119, 0.005098, 0.006063, 0.008031, 0.009843,
    0.011060, 0.011429, 0.012542, 0.014117, 0.014907, 0.016057,
    0.016994, 0.016994, 0.017586, 0.018334, 0.018951, 0.020098,
    0.021114, 0.022051, 0.022022, 0.022157, 0.023038, 0.023906,
    0.024438, 0.024267, 0.024061
]

beta_min = [
    0.497504, 0.495881, 0.494902, 0.493937, 0.491969, 0.490157,
    0.488940, 0.488571, 0.487458, 0.485883, 0.485093, 0.483943,
    0.483006, 0.483006, 0.482414, 0.481666, 0.481049, 0.479902,
    0.478886, 0.477949, 0.477978, 0.477843, 0.476962, 0.476094,
    0.475562, 0.475733, 0.475939
]

beta_max = [
    0.502056, 0.503156, 0.503874, 0.505169, 0.505283, 0.506072,
    0.507459, 0.507742, 0.508461, 0.509973, 0.511596, 0.512966,
    0.514186, 0.514908, 0.515181, 0.515615, 0.515680, 0.515265,
    0.515773, 0.516459, 0.517477, 0.518195, 0.518764, 0.519560,
    0.520343, 0.520466, 0.521244
]

# -------------------------
# Helper
# -------------------------
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

# -------------------------
# 1. Validation loss comparison
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, fixed_val, marker="o", label="Fixed β")
plt.plot(epochs, learn_val, marker="o", label="Learnable β")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss: Fixed β vs Learnable β")
plt.legend()
plt.grid(True)
save_plot("val_loss_comparison.png")

# -------------------------
# 2. Training loss comparison
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, fixed_train, marker="o", label="Fixed β")
plt.plot(epochs, learn_train, marker="o", label="Learnable β")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss: Fixed β vs Learnable β")
plt.legend()
plt.grid(True)
save_plot("train_loss_comparison.png")

# -------------------------
# 3. Generalization gap
# -------------------------
fixed_gap = [v - t for v, t in zip(fixed_val, fixed_train)]
learn_gap = [v - t for v, t in zip(learn_val, learn_train)]

plt.figure(figsize=(8, 5))
plt.plot(epochs, fixed_gap, marker="o", label="Fixed β")
plt.plot(epochs, learn_gap, marker="o", label="Learnable β")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss - Training Loss")
plt.title("Generalization Gap")
plt.legend()
plt.grid(True)
save_plot("generalization_gap.png")

# -------------------------
# 4. Validation improvement from epoch 25
# -------------------------
fixed_val_improve = [fixed_val[0] - v for v in fixed_val]
learn_val_improve = [learn_val[0] - v for v in learn_val]

plt.figure(figsize=(8, 5))
plt.plot(epochs, fixed_val_improve, marker="o", label="Fixed β")
plt.plot(epochs, learn_val_improve, marker="o", label="Learnable β")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss Improvement")
plt.title("Validation Improvement Since Epoch 25")
plt.legend()
plt.grid(True)
save_plot("val_improvement.png")

# -------------------------
# 5. Difference in validation loss
# positive means learnable beta is better
# -------------------------
val_advantage = [f - l for f, l in zip(fixed_val, learn_val)]

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_advantage, marker="o")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Epoch")
plt.ylabel("Fixed Val Loss - Learnable Val Loss")
plt.title("Validation Advantage of Learnable β")
plt.grid(True)
save_plot("learnable_beta_val_advantage.png")

# -------------------------
# 6. Better beta movement plot: beta mean shift
# -------------------------
beta_mean_shift = [b - 0.5 for b in beta_mean]
rwkv_shift = [b - 0.5 for b in rwkv_mean]
rffn_shift = [b - 0.5 for b in rffn_mean]

plt.figure(figsize=(8, 5))
plt.plot(epochs, beta_mean_shift, marker="o", label="Overall mean β - 0.5")
plt.plot(epochs, rwkv_shift, marker="o", label="RWKV mean β - 0.5")
plt.plot(epochs, rffn_shift, marker="o", label="RFFN mean β - 0.5")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Epoch")
plt.ylabel("Deviation from 0.5")
plt.title("Mean β Shift")
plt.legend()
plt.grid(True)
save_plot("beta_mean_shift.png")

# -------------------------
# 7. Beta spread
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, beta_std, marker="o", label="β standard deviation")
plt.plot(epochs, beta_max_dev, marker="o", label="Max |β - 0.5|")
plt.xlabel("Epoch")
plt.ylabel("Deviation")
plt.title("Spread of Learned β Values")
plt.legend()
plt.grid(True)
save_plot("beta_spread.png")

# -------------------------
# 8. Beta min/max range
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, beta_min, marker="o", label="Min β")
plt.plot(epochs, beta_max, marker="o", label="Max β")
plt.plot(epochs, beta_mean, marker="o", label="Mean β")
plt.axhline(0.5, linestyle="--", linewidth=1, label="Initial β = 0.5")
plt.xlabel("Epoch")
plt.ylabel("β Value")
plt.title("Range of Learned β Values")
plt.legend()
plt.grid(True)
save_plot("beta_range.png")

# -------------------------
# 9. Beta std vs validation loss
# -------------------------
plt.figure(figsize=(8, 5))
plt.scatter(beta_std, learn_val)
for e, x, y in zip(epochs, beta_std, learn_val):
    plt.annotate(str(e), (x, y), fontsize=8)
plt.xlabel("β Standard Deviation")
plt.ylabel("Learnable β Validation Loss")
plt.title("β Specialization vs Validation Loss")
plt.grid(True)
save_plot("beta_std_vs_val_loss.png")

# -------------------------
# Summary
# -------------------------
best_fixed_idx = min(range(len(fixed_val)), key=lambda i: fixed_val[i])
best_learn_idx = min(range(len(learn_val)), key=lambda i: learn_val[i])

print(f"Saved graphs to: {save_dir}")
print(f"Best fixed β val loss: {fixed_val[best_fixed_idx]:.4f} at epoch {epochs[best_fixed_idx]}")
print(f"Best learnable β val loss: {learn_val[best_learn_idx]:.4f} at epoch {epochs[best_learn_idx]}")
print(f"Final fixed β val loss: {fixed_val[-1]:.4f}")
print(f"Final learnable β val loss: {learn_val[-1]:.4f}")
print(f"Final mean β: {beta_mean[-1]:.6f}")
print(f"Final RWKV mean β: {rwkv_mean[-1]:.6f}")
print(f"Final RFFN mean β: {rffn_mean[-1]:.6f}")
print(f"Final β std: {beta_std[-1]:.6f}")
print(f"Final max |β - 0.5|: {beta_max_dev[-1]:.6f}")