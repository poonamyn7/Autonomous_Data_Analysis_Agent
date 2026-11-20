"""Step 7 - sanity-check the results and warn about anything unreliable."""

from modules.agent import MIN_ROWS_FOR_ML


def validate_results(profile, eda_results, ml_results, anomaly_count, plan):
    """Return ``{"warnings": [...]}``; an empty list means everything looks fine."""
    warnings = []
    rows = profile.get("rows", 0)

    if rows < MIN_ROWS_FOR_ML:
        warnings.append(
            f"Small dataset: only {rows} rows. Statistics, correlations and anomaly "
            "detection are less reliable with so little data."
        )

    for col, n_missing in profile.get("missing_before_cleaning", {}).items():
        if rows and n_missing / rows > 0.2:
            warnings.append(
                f"Column '{col}' had {n_missing / rows * 100:.0f}% missing values before "
                "cleaning; filled values may bias the results."
            )

    if not profile.get("numeric_columns"):
        warnings.append("No numeric columns were found, so numeric analysis could not be done.")

    if eda_results is not None:
        best = eda_results.get("strongest_correlation")
        if best is not None and abs(best[2]) >= 0.95:
            warnings.append(
                f"'{best[0]}' and '{best[1]}' are almost perfectly correlated "
                f"({best[2]:.3f}). Check that one is not just a copy or direct "
                "calculation of the other."
            )

    if ml_results is not None:
        if ml_results["r2"] < 0:
            warnings.append(
                "The model's R² is negative: it predicts worse than simply using the average."
            )
        elif ml_results["r2"] < 0.5:
            warnings.append(
                f"The model fit is weak (R² = {ml_results['r2']:.2f}); do not rely on its predictions."
            )
        if ml_results["r2"] > 0.99:
            warnings.append(
                "The model fit is almost perfect, which can mean data leakage "
                "(a feature that already contains the answer)."
            )
        if ml_results.get("n_test", 100) < 10:
            warnings.append("The test set has fewer than 10 rows, so the score is unstable.")

    if rows and anomaly_count / rows > 0.2:
        warnings.append(
            f"{anomaly_count} of {rows} rows were flagged as anomalies (over 20%); "
            "the data may be naturally varied rather than containing true outliers."
        )

    return {"warnings": warnings}
