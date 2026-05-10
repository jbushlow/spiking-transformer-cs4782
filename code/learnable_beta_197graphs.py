import os
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(project_root, "results", "learnable_beta_results_epoch197")
os.makedirs(save_dir, exist_ok=True)

# Compare only completed epochs available in BOTH runs
epochs = list(range(198, 204))

# -------------------------
# Fixed beta / regular run
# -------------------------
fixed_train = [
    1.1304, 1.1237, 1.1167, 1.1255, 1.1179, 1.1112
]

fixed_val = [
    1.0769, 1.0777, 1.0834, 1.0820, 1.0691, 1.0722
]

# -------------------------
# Learnable beta run
# -------------------------
learn_train = [
    1.1438, 1.1297, 1.1217, 1.1299, 1.1222, 1.1150
]

learn_val = [
    1.0712, 1.0766, 1.0826, 1.0755, 1.0678, 1.0712
]

# -------------------------
# Learnable beta stats
# -------------------------
beta_mean = [
    0.500059, 0.500073, 0.500082,
    0.500085, 0.500080, 0.500078
]

rwkv_mean = [
    0.500036, 0.500019, 0.499995,
    0.499965, 0.499934, 0.499909
]

rffn_mean = [
    0.500081, 0.500127, 0.500170,
    0.500205, 0.500226, 0.500246
]

beta_min = [
    0.498742, 0.498168, 0.497380,
    0.497115, 0.496743, 0.496193
]

beta_max = [
    0.501642, 0.502237, 0.503089,
    0.503393, 0.503819, 0.504273
]

beta_std = [
    0.000352, 0.000473, 0.000588,
    0.000700, 0.000802, 0.000898
]

beta_max_dev = [
    0.001642, 0.002237, 0.003089,
    0.003393, 0.003819, 0.004273
]


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
# 4. Validation improvement from epoch 198
# -------------------------
fixed_val_improve = [fixed_val[0] - v for v in fixed_val]
learn_val_improve = [learn_val[0] - v for v in learn_val]

plt.figure(figsize=(8, 5))
plt.plot(epochs, fixed_val_improve, marker="o", label="Fixed β")
plt.plot(epochs, learn_val_improve, marker="o", label="Learnable β")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss Improvement")
plt.title("Validation Improvement Since Epoch 198")
plt.legend()
plt.grid(True)
save_plot("val_improvement.png")


# -------------------------
# 5. Validation advantage of learnable beta
# Positive means learnable beta is better
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
# 6. Mean beta shift
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
