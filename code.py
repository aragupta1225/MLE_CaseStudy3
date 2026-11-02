"""Case Study 3: E-Commerce Customer Segmentation (AI201B CA2).

Install (Python 3.11+): pip install pandas numpy scikit-learn matplotlib
Run: python code.py              # reads dataset.csv and creates output.pdf

Data: D. Chen, Online Retail, UCI, https://doi.org/10.24432/C5BW33
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
dataset.csv contains all original rows, keeping the five columns needed for RFM.
Original workbook: https://archive.ics.uci.edu/static/public/352/online+retail.zip
No synthetic transactions are used. Monetary values are in GBP.
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")                     # Also works without a display.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import sklearn

SEED = 42
FOLDER = Path(__file__).resolve().parent
FEATURES = ["Recency", "Frequency", "Monetary"]


def make_rfm(data):
    """One row per customer: days since last purchase, orders, total spend."""
    data = data.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
    cancelled = data["InvoiceNo"].astype(str).str.upper().str.startswith("C")
    data = data[~cancelled & (data["Quantity"] > 0) & (data["UnitPrice"] > 0)].copy()
    data["CustomerID"] = data["CustomerID"].astype(int)
    data["Amount"] = data["Quantity"] * data["UnitPrice"]
    snapshot = data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = data.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (snapshot - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),  # Orders, not number of product rows.
        Monetary=("Amount", "sum"),
    )
    return rfm, data, snapshot


def silhouette(points, labels):
    """Exclude DBSCAN noise (-1); a score needs at least two clusters."""
    keep = labels != -1
    if 2 <= len(set(labels[keep])) < keep.sum():
        return silhouette_score(points[keep], labels[keep],
                                sample_size=min(2000, keep.sum()), random_state=SEED)
    return np.nan


def text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=17, weight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        lines = textwrap.fill(str(paragraph), width=88)
        fig.text(0.08, y, lines, fontsize=10, va="top", linespacing=1.45)
        y -= 0.025 * (lines.count("\n") + 1) + 0.023
    fig.text(0.08, 0.035, "AI201B | CA2 | Case Study 3", fontsize=9, color="gray")
    pdf.savefig(fig)
    plt.close(fig)


def cluster_plot(axis, points, labels, title):
    for label in sorted(set(labels)):
        keep = labels == label
        name = "Noise" if label == -1 else f"Cluster {label}"
        axis.scatter(points[keep, 0], points[keep, 1], s=8, alpha=0.6,
                     label=f"{name} ({keep.sum()})", c="gray" if label == -1 else None)
    axis.set_title(title)
    axis.legend(fontsize=7, loc="best")


def persona(group, all_customers):
    """Business interpretation comes from median RFM, not cluster numbers."""
    r, f, m = group[FEATURES].median()
    low, high = all_customers[FEATURES].quantile([0.25, 0.75]).to_numpy()
    if r >= high[0]:
        return "Lapsed customers", "Test a win-back message; offer a relevant reminder."
    if f >= high[1] and m >= high[2]:
        return "Loyal high-value customers", "Offer early access and loyalty benefits."
    if r <= low[0] and f <= all_customers.Frequency.median():
        return "Recent occasional buyers", "Send onboarding tips and a second-purchase offer."
    if m >= high[2]:
        return "High-spend buyers", "Recommend complementary products and premium bundles."
    return "Occasional customers", "Test relevant cross-sells and replenishment reminders."


def main():
    # 1. Clean the real retailer's transactions and calculate RFM.
    raw = pd.read_csv(FOLDER / "dataset.csv", dtype={"InvoiceNo": str})
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    rfm, clean, snapshot = make_rfm(raw)
    print(f"Raw rows: {len(raw):,}; valid purchases: {len(clean):,}; customers: {len(rfm):,}")

    # 2. Log reduces skew; scaling gives the three features equal units.
    scaled = StandardScaler().fit_transform(np.log1p(rfm[FEATURES]))
    pca = PCA(n_components=2, random_state=SEED)
    reduced = pca.fit_transform(scaled)

    # 3. Choose K using silhouette; a higher value means clearer separation.
    k_values = list(range(2, 7))
    scores = []
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(reduced)
        scores.append(silhouette(reduced, labels))
    best_k = k_values[int(np.argmax(scores))]
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(reduced)

    # 4. DBSCAN finds dense groups and labels isolated customers as noise.
    # These are fixed, readable parameters; inspect sensitivity before business use.
    dbscan = DBSCAN(eps=0.15, min_samples=10).fit_predict(reduced)
    rfm["KMeans"] = kmeans
    rfm["DBSCAN"] = dbscan
    noise = np.mean(dbscan == -1)
    print(f"Chosen K: {best_k}; DBSCAN clusters: {len(set(dbscan) - {-1})}; noise: {noise:.1%}")

    # 5. t-SNE is for visualization only. Both plots use identical coordinates.
    embedding = TSNE(n_components=2, perplexity=30, init="pca",
                     learning_rate="auto", random_state=SEED).fit_transform(scaled)

    # 6. Save all results and interpretation in one output file.
    with PdfPages(FOLDER / "output.pdf") as pdf:
        text_page(pdf, "Case Study 3: Customer Segmentation", [
            "Aim: group retailer customers using Recency, Frequency and Monetary value; "
            "compare K-Means and DBSCAN, reduce dimensions with PCA, visualize with "
            "t-SNE, and propose actionable personas.",
            "Source: D. Chen (2015), Online Retail, UCI Machine Learning Repository. "
            "DOI: 10.24432/C5BW33. License: CC BY 4.0. "
            "https://archive.ics.uci.edu/dataset/352/online+retail",
            f"Included data: {len(raw):,} original transaction lines, five RFM columns. "
            f"Dates: {raw.InvoiceDate.min().date()} to {raw.InvoiceDate.max().date()}. "
            "No customer subsampling; all valid identified customers are included.",
            f"Cleaning retained {len(clean):,} rows and {len(rfm):,} customers; "
            f"removed {len(raw) - len(clean):,} rows with missing required values, "
            "canceled invoices, or nonpositive quantity/price. Original line items "
            "are preserved, including repeated purchases. Refunds are excluded: "
            "Monetary represents gross valid purchase value, not net revenue.",
            f"Snapshot: {snapshot.date()}. Recency = whole days since the latest purchase; "
            "Frequency = distinct invoice count; Monetary = sum(Quantity * UnitPrice), GBP.",
            "Workflow: clean -> RFM -> log1p -> standardize -> PCA (2D) -> both clustering "
            "algorithms. t-SNE uses the standardized three RFM features only for plots.",
            f"Run: pip install pandas numpy scikit-learn matplotlib; python code.py. "
            f"Seed: {SEED}. pandas {pd.__version__}; scikit-learn {sklearn.__version__}; "
            f"NumPy {np.__version__}; Matplotlib {matplotlib.__version__}.",
        ])
        fig, axes = plt.subplots(1, 3, figsize=(11.7, 5))
        for axis, feature in zip(axes, FEATURES):
            axis.hist(np.log1p(rfm[feature]), bins=35, color="steelblue")
            axis.set_title(feature)
            axis.set_xlabel(f"log(1 + {feature})")
            axis.set_ylabel("Customers")
        fig.suptitle("RFM distributions after log transformation")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.7, 5))
        axes[0].plot(k_values, scores, "o-")
        axes[0].set(xlabel="Number of K-Means clusters", ylabel="Silhouette score",
                    title=f"K selection: {best_k} chosen", xticks=k_values)
        axes[1].bar(["PC1", "PC2"], pca.explained_variance_ratio_)
        axes[1].set(ylabel="Fraction of variance", title="PCA explained variance")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for points, name in [(reduced, "PCA"), (embedding, "t-SNE")]:
            fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.5))
            cluster_plot(axes[0], points, kmeans, f"{name}: K-Means")
            cluster_plot(axes[1], points, dbscan, f"{name}: DBSCAN")
            for axis in axes:
                axis.set(xlabel=f"{name} dimension 1", ylabel=f"{name} dimension 2")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        text_page(pdf, "Comparison and limitations", [
            f"PCA retains {pca.explained_variance_ratio_.sum():.1%} of standardized "
            "log-RFM variance. Clusters and silhouette scores use these two PCs.",
            f"K-Means: K={best_k}; silhouette={silhouette(reduced, kmeans):.3f}. "
            "Every customer receives a cluster. K-Means favors compact groups and is "
            "sensitive to scaling, extreme customers and the chosen K.",
            f"DBSCAN: eps=0.15, min_samples=10; clusters={len(set(dbscan) - {-1})}; "
            f"noise={noise:.1%}; non-noise silhouette={silhouette(reduced, dbscan):.3f}. "
            "Noise is excluded from its silhouette, so that score cannot be compared "
            "directly with K-Means without considering coverage. NaN means fewer than "
            "two usable clusters. Noise means unusual purchase behavior, not fraud.",
            f"K-Means labels evaluated in all three standardized RFM dimensions: "
            f"silhouette={silhouette(scaled, kmeans):.3f}. DBSCAN non-noise labels in "
            f"three dimensions: silhouette={silhouette(scaled, dbscan):.3f}. "
            "This checks whether separation also appears outside the 2D projection.",
            "t-SNE: perplexity=30, seed=42. Nearby points can reveal local structure; "
            "global distances and apparent island sizes are not reliable evidence of "
            "a segment. Neither model is fitted on t-SNE coordinates.",
            "Limitations: one historical retailer, gross spend, no demographic or "
            "profitability data. K selection uses a fixed silhouette sample of up to "
            "2,000 customers. DBSCAN uses a fixed radius; sensitivity is shown below. "
            "Personas are median-based hypotheses, not measured campaign results. "
            "Validate actions using consented A/B tests and monitor segments over time.",
            "Methods: https://scikit-learn.org/stable/modules/clustering.html; "
            "https://scikit-learn.org/stable/modules/decomposition.html#pca; "
            "https://scikit-learn.org/stable/modules/manifold.html#t-sne",
        ])

        # Split persona pages into small groups to keep the PDF readable.
        for method in ["KMeans", "DBSCAN"]:
            paragraphs = []
            for label, group in rfm.groupby(method):
                title, action = persona(group, rfm)
                if label == -1:
                    title, action = "Unusual customers / noise", "Inspect individually before sending mass campaigns."
                r, f, m = group[FEATURES].median()
                paragraphs.append(
                    f"Cluster {label}: {title}. Customers: {len(group):,} "
                    f"({len(group)/len(rfm):.1%}). Median R={r:.0f} days, F={f:.0f} "
                    f"orders, M=GBP {m:,.2f}. Action: {action}")
            for start in range(0, len(paragraphs), 6):
                text_page(pdf, f"{method}: personas and marketing actions", paragraphs[start:start+6])

        sensitivity = []
        for eps in [0.10, 0.15, 0.20, 0.30]:
            labels = DBSCAN(eps=eps, min_samples=10).fit_predict(reduced)
            sensitivity.append(f"eps={eps:.2f}: {len(set(labels) - {-1})} clusters; "
                               f"noise={np.mean(labels == -1):.1%}; "
                               f"non-noise silhouette={silhouette(reduced, labels):.3f}.")
        text_page(pdf, "DBSCAN radius sensitivity", sensitivity + [
            "A small radius can fragment groups and mark many customers as noise; "
            "a larger radius can join groups. Choose settings with business review, "
            "coverage and stability checks, rather than selecting the largest silhouette alone."])
    print("Created output.pdf")


if __name__ == "__main__":
    main()
