import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.lines as mlines


# from firstclassifier.py

def fanoFac(spikes, bin_width, smooth=None, T=1000):
    edges = np.arange(0, T + 1, bin_width)
    binned = np.add.reduceat(spikes, edges[:-1], axis=1)
    mean = binned.mean(axis=0)
    var = binned.var(axis=0)
    epsilon = 1e-10
    safe_mean = np.where(mean < epsilon, 0, mean)
    fano = np.divide(var, safe_mean, out=np.zeros_like(var), where=safe_mean != 0)
    win = smooth // bin_width if smooth else None
    if smooth is not None:
        kernel = np.ones(int(win)) / win
        fano = convolve(fano, kernel, mode='same')
    times = edges[:-1] + bin_width / 2
    return times, fano


def genFanos(dataset, datasize, T=1000, bw=20):
    fanos = np.array([
        fanoFac(dataset[i], bin_width=bw, T=T)[1] for i in range(datasize * 2)
    ])
    n = len(fanos[0])
    clip_fac = 0.1
    fanos = fanos[:, int(np.floor(n * clip_fac)) : int(np.ceil(n * (1 - clip_fac)))]
    fanos = gaussian_filter1d(fanos, sigma=1, axis=1)
    return np.nan_to_num(fanos, nan=0.0)



def maxFanoFac(fanos, threshold=1.22):
    f_max = np.max(fanos, axis=1)
    return (f_max > threshold).astype(int)


# sub‑classifier 1 – max Fano factor

def genFanoClassifyMax(dataset, datasize, T=1000, bw=20, threshold=1.22):
    fanos = np.array([
        fanoFac(dataset[i], bin_width=bw, T=T)[1] for i in range(datasize * 2)
    ])
    n = len(fanos[0])
    clip_fac = 0.15
    fanos = fanos[:, int(np.floor(n * clip_fac)) : int(np.ceil(n * (1 - clip_fac)))]
    fanos = gaussian_filter1d(fanos, sigma=1, axis=1)
    f_max = np.max(fanos, axis=1)
    classifier = (f_max > threshold).astype(int)
    return classifier


# sub‑classifier 2 – max derivative of Fano

def maxFanoDeriv(dataset, datasize, T=1000, bw=20, threshold=0.002):
    fanos = np.array([
        fanoFac(dataset[i], bin_width=bw, smooth=100, T=T)[1]
        for i in range(datasize * 2)
    ])
    times = np.array(
        fanoFac(dataset[0], bin_width=bw, smooth=100, T=T)[0]
    )
    n = len(fanos[0])
    clip_fac = 0.1
    fanos = fanos[:, int(np.floor(n * clip_fac)) : int(np.ceil(n * (1 - clip_fac)))]
    times = times[int(np.floor(n * clip_fac)) : int(np.ceil(n * (1 - clip_fac)))]
    fanos_smoothed = np.array([
        gaussian_filter1d(fanos[i], sigma=3) for i in range(datasize * 2)
    ])
    gradients = [np.gradient(fanos_smoothed[i], times) for i in range(datasize * 2)]
    g_max = np.max(np.abs(gradients), axis=1)
    classifier = (g_max > threshold).astype(int)
    return g_max, classifier


# sub‑classifier 3 – LoG range

def rangeLoG(dataset, datasize, T=1000, bw=20, threshold=0.001):
    fanos = np.array([
        fanoFac(dataset[i], bin_width=bw, smooth=100, T=T)[1]
        for i in range(datasize * 2)
    ])
    n = len(fanos[0])
    clip_frac = 0.1
    fanos = fanos[:, int(np.floor(n * clip_frac)) : int(np.ceil(n * (1 - clip_frac)))]
    LoG_responses = [gaussian_laplace(fanos[i], sigma=8) for i in range(datasize * 2)]
    lg_max = np.max(LoG_responses, axis=1)
    lg_min = np.min(LoG_responses, axis=1)
    rg = lg_max - lg_min
    classifier = (rg > threshold).astype(int)
    return rg, classifier


# metric helpers

def accuracy(predictions, num):
    ground_truth = np.array([0] * num + [1] * num)
    correct = predictions == ground_truth
    acc = correct.mean()
    return acc, correct


# majority‑vote ensemble


def fanoClassify(dataset, datasize, T=1000, bw=20):
    mff = genFanoClassifyMax(dataset, datasize)
    _, mfd = maxFanoDeriv(dataset, datasize)
    _, lgc = rangeLoG(dataset, datasize)
    cls = np.vstack([mff, mfd, lgc])            # shape (3, 2*datasize)
    prds = np.where(cls == 0, -1, 1)            # vote −1 (step) or +1 (ramp)
    sum_prds = np.sum(prds, axis=0)
    classifier = (sum_prds > 0).astype(int)     # majority vote → ramp?
    performance, _ = accuracy(classifier, datasize)
    disagreements = np.sum((mff != mfd) | (mff != lgc) | (mfd != lgc))
    return classifier, performance, disagreements




# Plotting

def _plot_confusion_and_roc(scores, y_true, y_pred, name="subclf"):
    """Confusion matrix + ROC for a single classifier."""
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["step", "ramp"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"{name}: confusion matrix")

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(3.8, 3.8))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False‑positive rate")
    plt.ylabel("True‑positive rate")
    plt.title(f"{name}: ROC curve")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.tight_layout()


def _plot_threshold_1d(scores, y_true, threshold, name="subclf"):
    """1‑D decision plot: score scatter with vertical threshold line."""
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.02, 0.02, size=len(scores))
    colours = np.array(["tab:blue", "tab:orange"])

    plt.figure(figsize=(6, 2.4))
    plt.scatter(scores, jitter, c=colours[y_true], edgecolor="k", alpha=0.8, s=38)
    plt.axvline(threshold, color="k", linewidth=2, label="threshold")
    plt.yticks([])
    plt.xlabel("score (larger → ramp)")
    plt.title(f"{name}: decision boundary (1‑D)")
    plt.legend()
    plt.tight_layout()


# Evaluation 

def evaluate_and_plot(dataset, datasize, T=1000, bw=20):

    y_true = np.array([0] * datasize + [1] * datasize)

    fanos = np.array([
        fanoFac(dataset[i], bin_width=bw, T=T)[1] for i in range(datasize * 2)
    ])
    n = fanos.shape[1]
    clip_fac = 0.15
    fanos_clipped = fanos[
        :, int(np.floor(n * clip_fac)) : int(np.ceil(n * (1 - clip_fac)))
    ]
    fanos_sm = gaussian_filter1d(fanos_clipped, sigma=1, axis=1)
    f_max = np.max(fanos_sm, axis=1)
    thr_mff = 1.22
    mff_pred = (f_max > thr_mff).astype(int)

    _plot_confusion_and_roc(f_max, y_true, mff_pred, name="max‑Fano")
    _plot_threshold_1d(f_max, y_true, thr_mff, name="max‑Fano")

    
    g_max, mfd_pred = maxFanoDeriv(dataset, datasize, T=T, bw=bw)
    thr_mfd = 0.002
    _plot_confusion_and_roc(g_max, y_true, mfd_pred, name="max‑dF/dt")
    _plot_threshold_1d(g_max, y_true, thr_mfd, name="max‑dF/dt")

    
    rg, lgc_pred = rangeLoG(dataset, datasize, T=T, bw=bw)
    thr_lgc = 0.001
    _plot_confusion_and_roc(rg, y_true, lgc_pred, name="LoG‑range")
    _plot_threshold_1d(rg, y_true, thr_lgc, name="LoG‑range")

    
    ensemble_pred, ensemble_acc, disagreements = fanoClassify(dataset, datasize, T=T, bw=bw)
    vote_score = (
        np.where(mff_pred == 0, -1, 1)
        + np.where(mfd_pred == 0, -1, 1)
        + np.where(lgc_pred == 0, -1, 1)
    )

    _plot_confusion_and_roc(vote_score, y_true, ensemble_pred, name="ensemble")

    print("\n=== Summary ===")
    for name, pred in zip(
        ["max‑Fano", "max‑dF/dt", "LoG‑range", "ensemble"],
        [mff_pred, mfd_pred, lgc_pred, ensemble_pred],
    ):
        acc = (pred == y_true).mean() * 100
        print(f"{name:<12}: accuracy = {acc:5.1f}%  ({(pred==y_true).sum()}/{len(y_true)})")

    print(f"Disagreements among the 3 votes: {disagreements}/{len(y_true)} samples")
    idx_dis = np.where((mff_pred != mfd_pred) | (mff_pred != lgc_pred) | (mfd_pred != lgc_pred))[0]
    print("Indices with disagreement (first 20 shown):", idx_dis[:20])

    return {
        "mff_pred": mff_pred,
        "mfd_pred": mfd_pred,
        "lgc_pred": lgc_pred,
        "ensemble_pred": ensemble_pred,
        "disagreements": idx_dis,
    }


def param_agreement_plot(
    ramp_params,
    step_params,
    mff_pred,
    mfd_pred,
    lgc_pred,
    ensemble_pred,
    y_true,
):
    """
    Scatter the beta-sigma (ramp) and m-r (step) spaces.
    Colour = ensemble correctness (green/red).
    Marker = unanimous dot vs  split cross.
    """

    N_ramp = ramp_params.shape[0]
    N_step = step_params.shape[0]

    # unanimous?  ensemble correct?
    unanimous = (mff_pred == mfd_pred) & (mff_pred == lgc_pred)
    correct   = ensemble_pred == y_true

    colours = np.where(correct, "tab:green", "tab:red")
    markers = np.where(unanimous, "o", "x")

    # split by model type
    col_ramp, col_step = colours[:N_ramp], colours[N_ramp:]
    mark_ramp, mark_step = markers[:N_ramp], markers[N_ramp:]

    def _scatter(ax, xs, ys, cols, mks):
        for xi, yi, c, m in zip(xs, ys, cols, mks):
            ax.scatter(xi, yi, c=c, marker=m,
                       edgecolor="k", s=70, alpha=0.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── ramp (beta vs sigma)
    beta, sigma = ramp_params.T
    _scatter(ax1, beta, sigma, col_ramp, mark_ramp)
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel(r"$\sigma$")
    ax1.set_title("Ramp parameter space")

    # ── step (m vs r)
    m_vals, r_vals = step_params.T
    _scatter(ax2, m_vals, r_vals, col_step, mark_step)
    ax2.set_xlabel("m (step time)")
    ax2.set_ylabel("r (rate ratio)")
    ax2.set_title("Step parameter space")

    # legend
    legend_elems = [
        mlines.Line2D([], [], color="tab:green", marker="o", ls="",
                      label="correct & unanimous"),
        mlines.Line2D([], [], color="tab:green", marker="x", ls="",
                      label="correct & split"),
        mlines.Line2D([], [], color="tab:red", marker="o", ls="",
                      label="wrong & unanimous"),
        mlines.Line2D([], [], color="tab:red", marker="x", ls="",
                      label="wrong & split"),
    ]
    ax2.legend(handles=legend_elems, loc="best", frameon=True)

    plt.tight_layout()

    # quick numeric summary
    ramp_acc = correct[:N_ramp].mean() * 100
    step_acc = correct[N_ramp:].mean() * 100
    ramp_uni = unanimous[:N_ramp].mean() * 100
    step_uni = unanimous[N_ramp:].mean() * 100
    print("\n=== Parameter-space summary ===")
    print(f"Ramp - accuracy: {ramp_acc:5.1f}%   unanimous: {ramp_uni:5.1f}%")
    print(f"Step - accuracy: {step_acc:5.1f}%   unanimous: {step_uni:5.1f}%")

    return dict(
        ramp_accuracy=ramp_acc,
        step_accuracy=step_acc,
        ramp_unanimous=ramp_uni,
        step_unanimous=step_uni,
    )

