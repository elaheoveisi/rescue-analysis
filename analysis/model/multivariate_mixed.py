import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def prepare_univariate_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Best run per participant/trial, with condition/expertise coded the
    same way glmm.run_all does (kept in sync with that function)."""
    condition_map = cfg.get("analysis", {}).get("condition_by_category", {})

    subjects = cfg.get("sub", [])
    if subjects:
        df = df[df["participant"].isin(subjects)]

    metric = cfg.get("glmm2", {}).get("best_run_metric", "saved_victims")
    df = (
        df.sort_values(metric, ascending=False)
        .groupby(["participant", "trial"], as_index=False)
        .first()
    )
    df = df.rename(columns={"trial": "category"})
    df["condition"] = df["category"].map(condition_map).fillna("unknown")
    df["condition"] = pd.Categorical(df["condition"], categories=["no_llm", "llm"])
    df["expertise"] = pd.Categorical(df["expertise"], categories=["novice", "expert"])
    return df


def _fixed_effect_rows(idata: az.InferenceData, outcomes: list[str]) -> pd.DataFrame:
    summary = az.summary(
        idata,
        var_names=["alpha", "beta_condition", "beta_expertise", "beta_interaction"],
        hdi_prob=0.95,
    )

    term_names = {
        "alpha": "Intercept",
        "beta_condition": "condition_llm",
        "beta_expertise": "expertise_expert",
        "beta_interaction": "condition_llm:expertise_expert",
    }

    rows = []
    for var_name, term_name in term_names.items():
        for idx, outcome in enumerate(outcomes):
            row_name = f"{var_name}[{idx}]"
            if row_name not in summary.index:
                continue
            stats = summary.loc[row_name]
            rows.append(
                {
                    "outcome": outcome,
                    "term": term_name,
                    "mean": stats["mean"],
                    "sd": stats["sd"],
                    "hdi_2.5%": stats["hdi_2.5%"],
                    "hdi_97.5%": stats["hdi_97.5%"],
                }
            )
    return pd.DataFrame(rows)


def fit_multivariate_mixed_model(
    df: pd.DataFrame,
    outcomes: list[str],
    cfg: dict,
) -> az.InferenceData | None:
    available = [outcome for outcome in outcomes if outcome in df.columns]
    if not available:
        return None

    clean = df.dropna(subset=available).copy()
    clean = clean[
        clean["condition"].isin(["no_llm", "llm"])
        & clean["expertise"].isin(["novice", "expert"])
    ]
    if clean.empty:
        return None

    participant_codes = pd.Categorical(clean["participant"])
    participant_idx = participant_codes.codes
    n_participants = len(participant_codes.categories)

    multivariate_cfg = cfg["glmm2"].get("multivariate", {})
    coding_cfg = multivariate_cfg.get("coding", {})
    condition_true = coding_cfg.get("condition_true", "llm")
    expertise_true = coding_cfg.get("expertise_true", "expert")

    condition_data = (
        clean["condition"].astype(str).eq(condition_true).astype(float).to_numpy()
    )
    expertise_data = (
        clean["expertise"].astype(str).eq(expertise_true).astype(float).to_numpy()
    )
    interaction_data = condition_data * expertise_data
    y = clean[available].astype(float).to_numpy()
    _, n_outcomes = y.shape

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2, shape=n_outcomes)
        beta_condition = pm.Normal("beta_condition", mu=0, sigma=2, shape=n_outcomes)
        beta_expertise = pm.Normal("beta_expertise", mu=0, sigma=2, shape=n_outcomes)
        beta_interaction = pm.Normal(
            "beta_interaction", mu=0, sigma=2, shape=n_outcomes
        )

        sigma_participant = pm.HalfNormal(
            "sigma_participant", sigma=1, shape=n_outcomes
        )
        participant_offset = pm.Normal(
            "participant_offset",
            mu=0,
            sigma=sigma_participant,
            shape=(n_participants, n_outcomes),
        )

        mu = (
            alpha
            + beta_condition * condition_data[:, None]
            + beta_expertise * expertise_data[:, None]
            + beta_interaction * interaction_data[:, None]
            + participant_offset[participant_idx]
        )
        sigma = pm.HalfNormal("sigma", sigma=1, shape=n_outcomes)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=multivariate_cfg.get("draws", 1000),
            tune=multivariate_cfg.get("tune", 1000),
            chains=multivariate_cfg.get("chains", 4),
            target_accept=multivariate_cfg.get("target_accept", 0.9),
            random_seed=multivariate_cfg.get("random_seed", 42),
            progressbar=False,
        )

    return idata


def run_all(cfg: dict, dataframes: dict) -> pd.DataFrame:
    """Run one multivariate mixed model per dataset."""
    outcomes = cfg["glmm2"]["continuous"] + cfg["glmm2"]["count"]
    rows = []

    for name, df in dataframes.items():
        if df is None or df.empty:
            continue

        prepared = prepare_univariate_df(df, cfg)
        prepared = prepared[
            ["participant", "category", "condition", "expertise"]
            + [o for o in outcomes if o in prepared.columns]
        ]
        idata = fit_multivariate_mixed_model(prepared, outcomes, cfg)
        if idata is None:
            continue

        summary = _fixed_effect_rows(
            idata, [o for o in outcomes if o in prepared.columns]
        )
        if summary.empty:
            continue

        summary.insert(0, "dataset", name)
        rows.append(summary)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "outcome",
                "term",
                "mean",
                "sd",
                "hdi_2.5%",
                "hdi_97.5%",
            ]
        )

    return pd.concat(rows, ignore_index=True)
