"""
Energy Consumption Pattern Clustering using K-Means
Machine Learning II (BAI702) - PBL, Problem Statement 14

Dataset : UCI "Individual household electric power consumption"
          One household near Paris, France, Dec 2006 - Nov 2010, 1-minute readings
          (2,075,259 rows). https://archive.ics.uci.edu/dataset/235
          License: CC BY 4.0 (open data)

Pipeline: load -> clean -> hourly mean -> daily 24-hour load profiles
          -> standardise -> feature-selection study -> choose k (elbow / silhouette /
          Davies-Bouldin / Calinski-Harabasz) -> final K-Means -> PCA view
          -> name & interpret clusters -> save figures, CSVs and a text report.

Run:   python energy_clustering.py                 (auto-downloads the dataset)
       python energy_clustering.py --data path/to/household_power_consumption.txt
       python energy_clustering.py --k 4           (force the number of clusters)
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

UCI_URL = ("https://archive.ics.uci.edu/static/public/235/"
           "individual+household+electric+power+consumption.zip")
RANDOM_STATE = 42
HOURS = list(range(24))
sns.set_theme(style="whitegrid", context="notebook")


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def get_dataset(path, data_dir):
    """Return path of household_power_consumption.txt (download if needed)."""
    if path:
        return path
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, "household_power_consumption.txt")
    if os.path.exists(target):
        return target
    print("Downloading UCI dataset (~20 MB zip) ...")
    try:
        raw = urllib.request.urlopen(UCI_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            z.extract("household_power_consumption.txt", data_dir)
        return target
    except Exception as e:  # noqa
        sys.exit(f"Could not download dataset ({e}).\nDownload the zip manually from "
                 "https://archive.ics.uci.edu/dataset/235 , extract it and run:\n"
                 "python energy_clustering.py --data household_power_consumption.txt")


def load_raw(path):
    df = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    format="%d/%m/%Y %H:%M:%S", errors="coerce")
    num = ["Global_active_power", "Global_reactive_power", "Voltage",
           "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df.drop(columns=["Date", "Time"])


# ----------------------------------------------------------------------------
# 2. CLEAN + HOURLY + DAILY PROFILES
# ----------------------------------------------------------------------------
def clean(df, log):
    log["raw_rows"] = len(df)
    df = df.dropna(subset=["datetime", "Global_active_power"])
    log["after_missing_removed"] = len(df)
    df = df[(df["Global_active_power"] >= 0) & (df["Global_active_power"] <= 25)]
    log["after_negative_and_gt25kW_removed"] = len(df)
    df = df.drop_duplicates(subset="datetime")
    log["after_duplicates_removed"] = len(df)
    return df.sort_values("datetime")


def hourly(df, log, min_readings=45):
    h = (df.set_index("datetime")["Global_active_power"]
           .resample("h").agg(["mean", "count"]))
    log["hours_total"] = len(h)
    h = h[h["count"] >= min_readings]["mean"]
    log["hours_kept(>=45 readings)"] = len(h)
    return h


def daily_profiles(h, log):
    t = h.to_frame("kW")
    t["date"] = t.index.normalize()
    t["hour"] = t.index.hour
    prof = t.pivot(index="date", columns="hour", values="kW").reindex(columns=HOURS)
    log["days_total"] = len(prof)
    prof = prof.interpolate(axis=1, limit=2, limit_direction="both")  # fill tiny gaps
    prof = prof.dropna()                                              # keep complete days
    log["days_complete_24h"] = len(prof)
    prof.columns = [f"h{c:02d}" for c in HOURS]
    return prof


# ----------------------------------------------------------------------------
# 3. FEATURE SETS (feature-selection study)
# ----------------------------------------------------------------------------
def build_feature_sets(prof):
    P = prof.values
    idx = prof.index
    weekend = (idx.dayofweek >= 5).astype(int)
    blocks = pd.DataFrame({
        "night_00_05": P[:, 0:6].mean(1), "morning_06_11": P[:, 6:12].mean(1),
        "afternoon_12_17": P[:, 12:18].mean(1), "evening_18_23": P[:, 18:24].mean(1)},
        index=idx)
    peak = P.argmax(1)
    summary = pd.DataFrame({
        "daily_mean": P.mean(1), "daily_max": P.max(1), "daily_std": P.std(1),
        "load_factor": P.mean(1) / P.max(1),
        "peak_hour_sin": np.sin(2 * np.pi * peak / 24),
        "peak_hour_cos": np.cos(2 * np.pi * peak / 24)}, index=idx)
    summary = pd.concat([summary, blocks], axis=1)
    hourly_wk = prof.copy()
    hourly_wk["is_weekend"] = weekend
    return {
        "A: 24 hourly values (main)": prof,
        "B: 4 time-block means": blocks,
        "C: summary statistics": summary,
        "D: 24 hourly + weekend flag": hourly_wk,
    }


def scale(X):
    return StandardScaler().fit_transform(X)


# ----------------------------------------------------------------------------
# 4. CHOOSING k
# ----------------------------------------------------------------------------
def evaluate_k(Xs, k_range):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        rows.append(dict(k=k, inertia=km.inertia_,
                         silhouette=silhouette_score(Xs, km.labels_),
                         davies_bouldin=davies_bouldin_score(Xs, km.labels_),
                         calinski_harabasz=calinski_harabasz_score(Xs, km.labels_)))
    return pd.DataFrame(rows)


def plot_k_selection(m, k_best, out):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    spec = [("inertia", "Elbow curve (inertia)", "Inertia (WCSS)"),
            ("silhouette", "Silhouette score (higher = better)", "Silhouette"),
            ("davies_bouldin", "Davies-Bouldin index (lower = better)", "DB index"),
            ("calinski_harabasz", "Calinski-Harabasz (higher = better)", "CH score")]
    for a, (c, t, y) in zip(ax, spec):
        a.plot(m["k"], m[c], "o-", lw=2)
        a.axvline(k_best, color="crimson", ls="--", label=f"chosen k = {k_best}")
        a.set(title=t, xlabel="Number of clusters k", ylabel=y)
        a.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 5. NAMING / INTERPRETATION
# ----------------------------------------------------------------------------
def name_clusters(centers_kw):
    """Rule-based names from each cluster's mean 24-h profile (kW)."""
    means = centers_kw.mean(axis=1)
    order = means.rank(method="first")
    k = len(means)
    names = {}
    for c in centers_kw.index:
        r = order[c]
        level = "Low" if r <= k / 3 else ("High" if r > 2 * k / 3 else "Medium")
        ph = int(np.argmax(centers_kw.loc[c].values))
        part = ("night" if ph < 5 else "morning" if ph < 11 else
                "midday" if ph < 17 else "evening")
        flat = centers_kw.loc[c].max() / centers_kw.loc[c].mean() < 1.6
        names[c] = f"C{c}: {level} usage, " + ("all-day flat" if flat else f"{part} peak")
    return names


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to household_power_consumption.txt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--k", type=int, help="force number of clusters")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = {}
    rep = []                                    # lines for the text report

    def say(s=""):
        print(s)
        rep.append(s)

    # ---- load, clean, profile
    path = get_dataset(a.data, a.data_dir)
    raw = load_raw(path)
    say("=" * 70)
    say("ENERGY CONSUMPTION PATTERN CLUSTERING (K-MEANS)")
    say("=" * 70)
    say(f"Dataset file : {os.path.basename(path)}")
    say(f"Period       : {raw['datetime'].min()}  ->  {raw['datetime'].max()}")
    df = clean(raw, log)
    h = hourly(df, log)
    prof = daily_profiles(h, log)
    say("\n[1] DATA PREPARATION")
    for k_, v_ in log.items():
        say(f"    {k_:<40}{v_:>12,}")
    say(f"    Final matrix: {prof.shape[0]} days x {prof.shape[1]} hourly features (kW)")
    say("\nDescriptive statistics of hourly load (kW):")
    say(h.describe().round(3).to_string())

    # ---- EDA plot: average profile weekday vs weekend
    fig, ax = plt.subplots(figsize=(9, 4))
    wk = prof.index.dayofweek >= 5
    ax.plot(HOURS, prof[~wk].mean(), "o-", label="Weekday average")
    ax.plot(HOURS, prof[wk].mean(), "s-", label="Weekend average")
    ax.set(title="Average daily load profile", xlabel="Hour of day", ylabel="Active power (kW)",
           xticks=range(0, 24, 2))
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{a.out}/01_average_daily_profile.png", dpi=150)
    plt.close()

    # ---- feature-selection study
    feats = build_feature_sets(prof)
    k_range = range(2, a.kmax + 3)
    Xmain = scale(feats["A: 24 hourly values (main)"])

    # ---- choose k on main features
    mk = evaluate_k(Xmain, k_range)
    cand = mk[(mk.k >= a.kmin) & (mk.k <= a.kmax)]
    k_best = int(cand.loc[cand.silhouette.idxmax(), "k"])
    K = a.k or k_best
    plot_k_selection(mk, K, f"{a.out}/02_choose_k.png")
    mk.round(4).to_csv(f"{a.out}/k_selection_metrics.csv", index=False)
    say("\n[2] CHOOSING THE NUMBER OF CLUSTERS (feature set A)")
    say(mk.round(4).to_string(index=False))
    say(f"\n    Best silhouette within k = {a.kmin}..{a.kmax}: k = {k_best}"
        + (f"   (overridden by --k -> using k = {K})" if a.k else ""))
    say("    (k = 2 usually gives the top silhouette because it only splits 'high' from "
        "'low' days,\n     so k >= 3 is searched to obtain more informative usage groups.)")

    # ---- feature-selection comparison at chosen K
    rows, labels_by_set = [], {}
    for name, F in feats.items():
        Xs = scale(F)
        km = KMeans(n_clusters=K, n_init=20, random_state=RANDOM_STATE).fit(Xs)
        labels_by_set[name] = km.labels_
        rows.append(dict(feature_set=name, n_features=F.shape[1],
                         silhouette=silhouette_score(Xs, km.labels_),
                         davies_bouldin=davies_bouldin_score(Xs, km.labels_),
                         calinski_harabasz=calinski_harabasz_score(Xs, km.labels_)))
    fs = pd.DataFrame(rows)
    ref = labels_by_set["A: 24 hourly values (main)"]
    fs["ARI_vs_A"] = [adjusted_rand_score(ref, labels_by_set[n]) for n in fs.feature_set]
    fs.round(4).to_csv(f"{a.out}/feature_selection_comparison.csv", index=False)
    say(f"\n[3] IMPACT OF FEATURE SELECTION (K = {K})")
    say(fs.round(4).to_string(index=False))
    say("    ARI_vs_A = agreement of the grouping with the main 24-hour profile "
        "(1 = identical).\n    Note: silhouette values from different feature spaces "
        "are indicative, not strictly comparable.")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a_, c, t in zip(ax, ["silhouette", "davies_bouldin", "ARI_vs_A"],
                        ["Silhouette (higher better)", "Davies-Bouldin (lower better)",
                         "Agreement with main grouping (ARI)"]):
        sns.barplot(data=fs, y="feature_set", x=c, ax=a_, color="steelblue")
        a_.set(title=t, ylabel="")
        if c != "silhouette":
            a_.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(f"{a.out}/03_feature_selection.png", dpi=150)
    plt.close()

    # ---- final model
    km = KMeans(n_clusters=K, n_init=30, random_state=RANDOM_STATE).fit(Xmain)
    lab = km.labels_
    prof_out = prof.copy()
    prof_out["cluster"] = lab
    sil = silhouette_score(Xmain, lab)
    db = davies_bouldin_score(Xmain, lab)
    ch = calinski_harabasz_score(Xmain, lab)
    ari = np.mean([adjusted_rand_score(lab, KMeans(K, n_init=10, random_state=s)
                                       .fit(Xmain).labels_) for s in range(1, 9)])
    say(f"\n[4] FINAL K-MEANS MODEL (k = {K}, features = 24 standardised hourly values)")
    say(f"    Silhouette score        : {sil:.4f}")
    say(f"    Davies-Bouldin index    : {db:.4f}")
    say(f"    Calinski-Harabasz score : {ch:.1f}")
    say(f"    Stability (mean ARI over 8 other random seeds): {ari:.3f}")

    # centres back in kW
    centers = prof_out.groupby("cluster")[prof.columns].mean()
    centers.columns = HOURS
    names = name_clusters(centers)
    nm = pd.Series(lab).map(names).values
    prof_out["cluster_name"] = nm

    # ---- PCA view
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(Xmain)
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", K)
    for c in range(K):
        m = lab == c
        ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=.65, color=palette[c], label=names[c])
    cz = pca.transform(km.cluster_centers_)
    ax.scatter(cz[:, 0], cz[:, 1], marker="X", s=250, c="black", edgecolor="white", label="Centroids")
    ax.set(title="K-Means clusters of daily load profiles (PCA 2-D view)",
           xlabel=f"PC1 ({ev[0]*100:.1f}% variance)", ylabel=f"PC2 ({ev[1]*100:.1f}% variance)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{a.out}/04_pca_clusters.png", dpi=150)
    plt.close()
    say(f"    PCA: PC1 + PC2 explain {100*ev.sum():.1f}% of the variance")

    # ---- mean profiles
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(K):
        ax.plot(HOURS, centers.loc[c], lw=2.4, color=palette[c], label=f"{names[c]}  (n={int((lab==c).sum())})")
    ax.set(title="Mean 24-hour load profile of each cluster", xlabel="Hour of day",
           ylabel="Active power (kW)", xticks=range(0, 24, 2))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{a.out}/05_cluster_profiles.png", dpi=150)
    plt.close()

    # ---- cluster sizes + weekend share + season
    idx = prof_out.index
    info = pd.DataFrame({"cluster": lab, "name": nm,
                         "weekend": idx.dayofweek >= 5, "month": idx.month,
                         "daily_kWh": prof.sum(axis=1).values}, index=idx)
    season = info.month.map(lambda m: "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5)
                            else "Summer" if m in (6, 7, 8) else "Autumn")
    info["season"] = season
    summ = info.groupby("cluster").agg(days=("cluster", "size"),
                                       avg_daily_kWh=("daily_kWh", "mean"),
                                       weekend_share=("weekend", "mean"))
    summ["share_%"] = 100 * summ["days"] / len(info)
    summ["peak_hour"] = centers.idxmax(axis=1)
    summ["peak_kW"] = centers.max(axis=1)
    summ["mean_kW"] = centers.mean(axis=1)
    summ.insert(0, "name", pd.Series(names))
    summ["weekend_share"] = (100 * summ["weekend_share"]).round(1)
    summ = summ.round(2)
    summ.to_csv(f"{a.out}/cluster_summary.csv")
    say("\n[5] CLUSTER INTERPRETATION")
    say(summ.to_string())
    say("\nSeason mix (% of each cluster's days):")
    sm = pd.crosstab(info.cluster, info.season, normalize="index").mul(100).round(1)
    say(sm.to_string())

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    sns.barplot(x=summ.index, y=summ["share_%"], ax=ax[0], hue=summ.index, palette=palette, legend=False)
    ax[0].set(title="Cluster size (% of days)", xlabel="Cluster", ylabel="% of days")
    sns.barplot(x=summ.index, y=summ["weekend_share"], ax=ax[1], hue=summ.index, palette=palette, legend=False)
    ax[1].set(title="Weekend days in each cluster (%)", xlabel="Cluster", ylabel="% weekend")
    ax[1].axhline(2 / 7 * 100, color="k", ls="--", lw=1, label="expected 28.6%")
    ax[1].legend()
    pd.crosstab(info.month, info.cluster, normalize="index").mul(100).plot(
        kind="bar", stacked=True, ax=ax[2], color=palette, width=.85)
    ax[2].set(title="Cluster mix by month of year", xlabel="Month", ylabel="% of days")
    ax[2].legend(title="Cluster", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{a.out}/06_cluster_size_weekend_season.png", dpi=150)
    plt.close()

    # heatmap of centres
    fig, ax = plt.subplots(figsize=(11, 0.9 * K + 1.8))
    sns.heatmap(centers.rename(index=names), cmap="YlOrRd", annot=False, ax=ax,
                cbar_kws={"label": "kW"})
    ax.set(title="Cluster centroids (kW) by hour of day", xlabel="Hour of day", ylabel="")
    plt.tight_layout()
    plt.savefig(f"{a.out}/07_centroid_heatmap.png", dpi=150)
    plt.close()

    prof_out.round(4).to_csv(f"{a.out}/daily_cluster_labels.csv")
    say("\n[6] FILES SAVED in " + a.out + "/")
    say("    figures 01-07 (.png), cluster_summary.csv, k_selection_metrics.csv,\n"
        "    feature_selection_comparison.csv, daily_cluster_labels.csv, report.txt")
    with open(f"{a.out}/report.txt", "w") as f:
        f.write("\n".join(rep))


if __name__ == "__main__":
    main()
