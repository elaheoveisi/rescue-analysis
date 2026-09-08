import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def run_glmm(df: pd.DataFrame, outcome: str) -> dict:
    """Fit a linear mixed-effects model with a participant random intercept."""
    model_df = (
        df[["participant", "condition", "expertise", outcome]]
        .dropna()
        .reset_index(drop=True)
    )
    if model_df.empty:
        return {"model_type": "skipped", "terms": []}

    formula = (
        f"{outcome} ~ C(condition, Treatment('no_llm'))"
        " * C(expertise, Treatment('novice'))"
    )
    model = smf.mixedlm(
        formula,
        model_df,
        groups=model_df["participant"].values,
        re_formula="1",
    ).fit()
    return {
        "model_type": "linear_mixedlm",
        "terms": [
            {
                "term": term,
                "coef": model.params[term],
                "se": model.bse.get(term),
                "p_value": model.pvalues.get(term),
            }
            for term in model.fe_params.index
        ],
    }


def run_all(cfg: dict, dataframes: dict) -> pd.DataFrame:
    """Run one mixed model per feature across all runs (participant random intercept handles repeated measures)."""
    condition_map = cfg.get("analysis", {}).get("condition_by_category", {})
    count_features = cfg["glmm2"]["count"]
    features = cfg["glmm2"]["continuous"] + count_features
    rows = []

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue

        subjects = cfg.get("sub", [])
        if subjects:
            df = df[df["participant"].isin(subjects)]
        if df.empty:
            continue

        metric = cfg.get("glmm2", {}).get("best_run_metric", "saved_victims")
        df = (
            df.sort_values(metric, ascending=False)
            .groupby(["participant", "trial"], as_index=False)
            .first()
        )
        df = df.rename(columns={"trial": "category"})
        df["condition"] = df["category"].map(condition_map).fillna("unknown")
        df["condition"] = pd.Categorical(df["condition"], categories=["no_llm", "llm"])
        df["expertise"] = pd.Categorical(
            df["expertise"], categories=["novice", "expert"]
        )

        keep = ["participant", "condition", "expertise"] + features
        prepared = df[keep].reset_index(drop=True)

        for col in count_features:
            if col in prepared.columns:
                prepared[col] = np.log1p(prepared[col])

        for outcome in features:
            result = run_glmm(prepared, outcome)
            rows.extend(
                {
                    "dataset": name,
                    "outcome": outcome,
                    "model_type": result["model_type"],
                    **term,
                }
                for term in result["terms"]
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        valid = results["p_value"].notna()
        _, fdr, _, _ = multipletests(results.loc[valid, "p_value"], method="fdr_bh")
        results["p_value_fdr"] = np.nan
        results.loc[valid, "p_value_fdr"] = fdr

    return results
