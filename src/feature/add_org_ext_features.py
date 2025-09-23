# src/feature/add_org_ext_features.py

# =========================================================================================================
# Add organizational and external context features that don’t exist in the base HR data
# =========================================================================================================
"""
Generates:

- GlassdoorRating (external employer attractiveness)
- JobMarketIndex (external job market pressure)
- TeamCohesion (internal cultural factor)
- NineBox, NineBoxScore (career alignment)

Note: Attrition generation is handled separately in control_randomness.py
"""

import numpy as np
import pandas as pd


class AddOrgExternalFeatures:
    """Class to enrich a dataset with synthetic organizational/external features."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate_glassdoor_rating(self, n: int) -> np.ndarray:
        """Simulate Glassdoor ratings between 3.5–4.0."""
        return self.rng.uniform(3.5, 4.0, size=n)

    def simulate_job_market_index(self, n: int) -> np.ndarray:
        """Simulate a synthetic job market index with sinusoidal variation."""
        years = np.linspace(0, 20, n)
        return (95 + 5 * np.sin(years / 3.0) + self.rng.normal(0, 1, size=n)).clip(85, 105)

    def simulate_team_cohesion(self, n: int) -> np.ndarray:
        """Simulate team cohesion scores between 40–100."""
        return self.rng.uniform(40, 100, size=n)

    def simulate_nine_box(self, n: int):
        """Simulate NineBox categories and scores."""
        labels = [f"Box_{i}" for i in range(1, 10)]
        scores = self.rng.integers(1, 10, size=n)
        nine_box = [labels[s - 1] for s in scores]
        return nine_box, scores

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add synthetic external/organizational features to a dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input HR dataset.

        Returns
        -------
        df : pd.DataFrame
            Enriched dataset with Glassdoor, Job Market, Team Cohesion, and NineBox features.
        """
        n = len(df)
        df["GlassdoorRating"] = self.simulate_glassdoor_rating(n)
        df["JobMarketIndex"] = self.simulate_job_market_index(n)
        df["TeamCohesion"] = self.simulate_team_cohesion(n)

        nine_box, nine_score = self.simulate_nine_box(n)
        df["NineBox"] = nine_box
        df["NineBoxScore"] = nine_score

        return df

# ================================================================================================
# End of src/feature/add_org_ext_features.py
# ================================================================================================