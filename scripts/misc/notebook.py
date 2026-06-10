# %%
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv("mock_evaluation_data.csv")
# %%
df
# %%
grouped = df.groupby(["Method", "Cost parameter", "Dataset"])
summary_df = pd.DataFrame(
    {
        "Avg. features chosen": grouped["Features chosen"].mean(),
        "Std. features chosen": grouped["Features chosen"].std(),
        "Avg. accuracy (external)": grouped["Accuracy (external)"].mean(),
        "Std. accuracy (external)": grouped["Accuracy (external)"].std(),
    }
).reset_index()
summary_df
# %%
fig, axes = plt.subplots(
    1,
    len(summary_df["Dataset"].unique()),
    figsize=(6 * len(summary_df["Dataset"].unique()), 5),
    sharey=True,
)

# %%
fg1 = sns.relplot(
    data=summary_df,
    kind="line",
    x="Avg. features chosen",
    y="Avg. accuracy (external)",
    hue="Method",
    col="Dataset",
    facet_kws={"sharey": False, "sharex": False},
).set(ylim=(0, 1))
for method, mdf in summary_df.groupby("Method"):
    ax.errorbar(
        mdf["Avg. features chosen"],
        mdf["Avg. accuracy (external)"],
        xerr=mdf["Std. features chosen"],
        yerr=mdf["Std. accuracy (external)"],
        fmt="none",
        ecolor="gray",
        alpha=0.5,
        capsize=2,
    )
fg1
# fg.figure.savefig("external_accuracy_vs_features_chosen.png", dpi=300)
# %% Same thing but for builtin classifier
grouped = df.dropna(subset=["Accuracy (builtin)"]).groupby(
    ["Method", "Cost parameter", "Dataset"]
)
summary_df = pd.DataFrame(
    {
        "Avg. features chosen": grouped["Features chosen"].mean(),
        "Std. features chosen": grouped["Features chosen"].std(),
        "Avg. accuracy (builtin)": grouped["Accuracy (builtin)"].mean(),
        "Std. accuracy (builtin)": grouped["Accuracy (builtin)"].std(),
    }
).reset_index()
# %%
fg2 = sns.relplot(
    data=summary_df,
    kind="scatter",
    x="Avg. features chosen",
    y="Avg. accuracy (builtin)",
    hue="Method",
    col="Dataset",
    facet_kws={"sharey": False, "sharex": False},
).set(ylim=(0, 1))
# fg.figure.savefig("builtin_accuracy_vs_features_chosen.png", dpi=300)

# %%
import matplotlib.pyplot as plt

# --- External classifier plot ---
fig, axes = plt.subplots(
    1,
    len(summary_df["Dataset"].unique()),
    figsize=(6 * len(summary_df["Dataset"].unique()), 5),
    sharey=True,
)
if len(summary_df["Dataset"].unique()) == 1:
    axes = [axes]

for ax, (dataset, subdf) in zip(
    axes, summary_df.groupby("Dataset"), strict=False
):
    sns.scatterplot(
        data=subdf,
        x="Avg. features chosen",
        y="Avg. accuracy (external)",
        hue="Method",
        marker="o",
        ax=ax,
    )
    for method, mdf in subdf.groupby("Method"):
        ax.errorbar(
            mdf["Avg. features chosen"],
            mdf["Avg. accuracy (external)"],
            xerr=mdf["Std. features chosen"],
            yerr=mdf["Std. accuracy (external)"],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=2,
        )
    ax.set_title(f"External - {dataset}")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# --- Builtin classifier plot ---
grouped = df.dropna(subset=["Accuracy (builtin)"]).groupby(
    ["Method", "Cost parameter", "Dataset"]
)
summary_df_builtin = pd.DataFrame(
    {
        "Avg. features chosen": grouped["Features chosen"].mean(),
        "Std. features chosen": grouped["Features chosen"].std(),
        "Avg. accuracy (builtin)": grouped["Accuracy (builtin)"].mean(),
        "Std. accuracy (builtin)": grouped["Accuracy (builtin)"].std(),
    }
).reset_index()

fig, axes = plt.subplots(
    1,
    len(summary_df_builtin["Dataset"].unique()),
    figsize=(6 * len(summary_df_builtin["Dataset"].unique()), 5),
    sharey=True,
)
if len(summary_df_builtin["Dataset"].unique()) == 1:
    axes = [axes]

for ax, (dataset, subdf) in zip(
    axes, summary_df_builtin.groupby("Dataset"), strict=False
):
    sns.scatterplot(
        data=subdf,
        x="Avg. features chosen",
        y="Avg. accuracy (builtin)",
        hue="Method",
        marker="o",
        ax=ax,
    )
    for method, mdf in subdf.groupby("Method"):
        ax.errorbar(
            mdf["Avg. features chosen"],
            mdf["Avg. accuracy (builtin)"],
            xerr=mdf["Std. features chosen"],
            yerr=mdf["Std. accuracy (builtin)"],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=2,
        )
    ax.set_title(f"Builtin - {dataset}")
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
fig.savefig("builtin_accuracy_vs_features_chosen_with_std.png", dpi=300)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

datasets = summary_df["Dataset"].unique()
n_datasets = len(datasets)

fig, axes = plt.subplots(
    2, n_datasets, figsize=(6 * n_datasets, 10), sharex=False, sharey=True
)

# --- External classifier (top row) ---
for i, dataset in enumerate(datasets):
    ax = axes[0, i] if n_datasets > 1 else axes[0]
    subdf = summary_df[summary_df["Dataset"] == dataset]
    sns.scatterplot(
        data=subdf,
        x="Avg. features chosen",
        y="Avg. accuracy (external)",
        hue="Method",
        marker="o",
        ax=ax,
        legend=(i == 0),  # Only show legend on first subplot
    )
    for method, mdf in subdf.groupby("Method"):
        ax.errorbar(
            mdf["Avg. features chosen"],
            mdf["Avg. accuracy (external)"],
            xerr=mdf["Std. features chosen"],
            yerr=mdf["Std. accuracy (external)"],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=2,
        )
    ax.set_title(f"External - {dataset}")
    ax.set_ylim(0, 1)

# --- Builtin classifier (bottom row) ---
datasets_builtin = summary_df_builtin["Dataset"].unique()
for i, dataset in enumerate(datasets_builtin):
    ax = axes[1, i] if n_datasets > 1 else axes[1]
    subdf = summary_df_builtin[summary_df_builtin["Dataset"] == dataset]
    sns.scatterplot(
        data=subdf,
        x="Avg. features chosen",
        y="Avg. accuracy (builtin)",
        hue="Method",
        marker="o",
        ax=ax,
        legend=False,  # Legend only on top row
    )
    for method, mdf in subdf.groupby("Method"):
        ax.errorbar(
            mdf["Avg. features chosen"],
            mdf["Avg. accuracy (builtin)"],
            xerr=mdf["Std. features chosen"],
            yerr=mdf["Std. accuracy (builtin)"],
            fmt="none",
            ecolor="gray",
            alpha=0.5,
            capsize=2,
        )
    ax.set_title(f"Builtin - {dataset}")
    ax.set_ylim(0, 1)

plt.tight_layout()
fig.savefig("combined_accuracy_vs_features_chosen_with_std.png", dpi=300)
plt.show()
