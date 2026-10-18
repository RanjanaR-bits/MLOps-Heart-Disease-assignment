"""Draws docs/architecture.png.  Usage: python scripts/make_architecture_diagram.py"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7)


# CI/CD band
ax.add_patch(FancyBboxPatch((0.2, 5.0), 12.6, 1.2, boxstyle="round,pad=0.05",
                            fc="#fff4d6", ec="#b8860b", lw=1.2))
ax.text(0.4, 5.95, "GitHub Actions CI/CD", fontsize=9, weight="bold", va="top")
for i, t in enumerate(["flake8 lint", "pytest unit tests", "train + MLflow",
                       "docker build + smoke test"]):
    box(1.3 + i * 3, 5.15, 2.4, 0.55, t, "#ffe9a8")
    if i:
        arrow(1.3 + i * 3 - 0.6, 5.42, 1.3 + i * 3, 5.42)

# ML flow
box(0.3, 2.9, 1.9, 1.1, "UCI Heart\nDisease data\n(src/download.py)", "#d9ecff")
box(2.7, 2.9, 1.9, 1.1, "Clean + EDA\n(src/data.py,\nsrc/eda.py)", "#d9ecff")
box(5.1, 2.9, 2.2, 1.1, "Pipeline: features\n+ impute/scale/encode\n+ LR / RF tuning", "#d9ecff")
box(7.8, 2.9, 1.7, 1.1, "model.joblib\n(best by CV AUC)", "#d4f5d4")
box(10.0, 2.9, 2.7, 1.1, "Docker image\nFastAPI /predict\n/health /metrics", "#d4f5d4")
arrow(2.2, 3.45, 2.7, 3.45)
arrow(4.6, 3.45, 5.1, 3.45)
arrow(7.3, 3.45, 7.8, 3.45)
arrow(9.5, 3.45, 10.0, 3.45)
box(5.1, 1.5, 2.2, 0.9, "MLflow tracking\n(params, metrics, plots,\nmodels)", "#f3d9ff")
arrow(6.2, 2.9, 6.2, 2.4)

# K8s
ax.add_patch(FancyBboxPatch((8.0, 0.2), 4.8, 2.2, boxstyle="round,pad=0.05",
                            fc="#eef", ec="#336", lw=1.2, ls="--"))
ax.text(8.2, 2.2, "Kubernetes (Minikube / Docker Desktop)", fontsize=8.5, weight="bold",
        va="top")
box(8.3, 0.4, 1.9, 1.1, "Service\n(LoadBalancer)", "#d4f5d4")
box(10.6, 0.4, 2.0, 1.1, "Deployment\n2 x API pods", "#d4f5d4")
arrow(10.2, 0.95, 10.6, 0.95)
arrow(11.6, 2.9, 11.6, 1.5, "deploy")
box(0.3, 0.55, 1.7, 0.9, "Client\n(curl / app)", "#eee")
arrow(2.0, 1.0, 8.3, 1.0, "POST /predict (JSON)")

# monitoring
box(5.1, 0.1, 1.9, 0.6, "Prometheus", "#ffd9d9")
box(2.8, 0.1, 1.9, 0.6, "Grafana", "#ffd9d9")
arrow(5.1, 0.4, 4.7, 0.4, "query")
arrow(7.0, 0.4, 8.3, 0.5, "scrape /metrics")

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=170)
print("saved docs/architecture.png")
