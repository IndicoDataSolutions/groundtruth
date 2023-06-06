#!/usr/bin/env python

import pandas as pd

THRESHOLDS = [0.85, 0.95, 0.99, 0.99999]


def get_metrics_df(
    df: pd.DataFrame, thresholds: list[float] = THRESHOLDS
) -> pd.DataFrame:
    df["accurate"] = df["accurate"].map({"TRUE": 1, "SHOULD_BE_TRUE": 1, "FALSE": 0})
    df.drop(df.loc[df["confidence"] == "__novalue__"].index, inplace=True)
    df["confidence"] = df["confidence"].astype(float)
    metrics = []
    for field in df["field"].unique():
        field_accuracy = {"Metric": "Accuracy", "Field": field}
        field_volume_count = {"Metric": "Volume Count", "Field": field}
        field_volume_percentage = {"Metric": "Volume Percentage", "Field": field}

        for threshold in thresholds:
            primary = df.loc[df["field"] == field]
            accuracy = primary.loc[primary["confidence"] >= threshold][
                "accurate"
            ].mean()

            total_count = len(primary)
            volume_count = len(primary.loc[primary["confidence"] >= threshold])
            volume_percentage = volume_count / total_count

            field_accuracy[f"{threshold:g}"] = accuracy
            field_volume_count[f"{threshold:g}"] = volume_count
            field_volume_percentage[f"{threshold:g}"] = volume_percentage

        metrics.append(field_accuracy)
        metrics.append(field_volume_count)
        metrics.append(field_volume_percentage)
    return pd.DataFrame(metrics).sort_values(["Metric", "Field"])


def compare_dfs(
    df_1: pd.DataFrame, df_2: pd.DataFrame, thresholds: list[float] = THRESHOLDS
) -> pd.DataFrame:
    df_all = pd.concat(
        [df_1.set_index(["Metric", "Field"]), df_2.set_index(["Metric", "Field"])],
        axis="columns",
        keys=["Original", "Resubmitted"],
    )
    df_final = df_all.swaplevel(axis="columns")[thresholds]
    return df_final


if __name__ == "__main__":
    extractions_1_df = pd.read_csv("./original_extractions.csv")
    extractions_2_df = pd.read_csv("./resubmitted_extractions.csv")

    df_1 = get_metrics_df(extractions_1_df)
    df_2 = get_metrics_df(extractions_2_df)

    df_1.to_csv("./original_metrics.csv", index=False)
    df_2.to_csv("./resubmitted_metrics.csv", index=False)

    df_all = compare_dfs(df_1, df_2)
    df_all.to_excel("./comparsion.xlsx")
