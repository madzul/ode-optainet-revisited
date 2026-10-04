"""Write the paper's figures (from the archived summaries) to figures/."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotting import fig_ccr, fig_ablation, fig_anytime

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    for name, fn, size in [("fig1_ccr", fig_ccr, (8, 5)), ("fig2_ablation", fig_ablation, (7, 4)),
                           ("fig3_anytime", fig_anytime, (8, 5))]:
        fig, ax = plt.subplots(figsize=size)
        fn(ax); fig.tight_layout(); fig.savefig(f"figures/{name}.png", dpi=150); plt.close(fig)
        print("wrote", f"figures/{name}.png")
