# ========================================================================
# control_randomeness.py
# ------------------------------------------------------------------------
# Purpose:
#   Generate a realistic outcomes using a probabilistic model.
#   Instead of directly mapping synthetic features -> attrition,
#   create a probabilistic risk model with controlled randomness.
#
# Key Idea:
#   Attrition is never fully deterministic in real life.
#   Instead, HR factors like Glassdoor rating, NineBox, Team Cohesion, etc.
#   increase or decrease the *probability* of attrition.
#
# Tunable Parameters:
#   - base_prob: natural baseline attrition rate
#   - hook_weights: dict of feature -> contribution
#   - interaction_weight: multiplier for combined low NineBox & low Cohesion
#   - noise_level: max uniform noise to add
#   - prob_cap: upper bound for attrition probability
#
# Input: 
#   - Base HR + Synthetic features
# Output:
#   - Adds `attrition_prob`: numeric probability of attrition per employee (latent risk)
#   - Adds `Attrition`: final binary outcome, sampled with randomness
#
# ========================================================================

import numpy as np
import pandas as pd

class RandomnessIntoDataset:

    def __init__(self,
                 base_prob: float = 0.05,
                 hook_weights: dict = None,
                 interaction_weight: float = 0.15,
                 noise_level: float = 0.05,
                 prob_cap: float = 0.9,
                 min_attrition_ratio: float = 0.05):
        """
        Parameters
        ----------
        base_prob : float
            Baseline attrition probability (default: 0.05 = 5%)
        hook_weights : dict
            Dict mapping feature -> probability increment.
            Defaults provided if None.
        interaction_weight : float
            Extra bump when (low NineBox & low Cohesion) both occur.
        noise_level : float
            Maximum uniform noise added per row (0 to noise_level).
        prob_cap : float
            Maximum probability cap (nobody has 100% certainty).
        min_attrition_ratio : float
            Minimum fraction of employees to mark as attrition (safeguard).
        """
        self.base_prob = base_prob
        self.hook_weights = hook_weights or {
            "GlassdoorRating": 0.10,   # poor Glassdoor
            "JobMarketIndex": 0.08,    # hot job market
            "NineBoxScore": 0.12,      # low career alignment
            "TeamCohesion": 0.10       # weak social cohesion
        }
        self.interaction_weight = interaction_weight
        self.noise_level = noise_level
        self.prob_cap = prob_cap
        self.min_attrition_ratio = min_attrition_ratio

    def generate_attrition(self, df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        """
        Create a probabilistic attrition column for HR dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with enriched HR features including synthetic ones:
            GlassdoorRating, JobMarketIndex, NineBoxScore, TeamCohesion, etc.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        df : pd.DataFrame
            Same DataFrame with two new columns:
            - 'attrition_prob' (float, 0-1)
            - 'Attrition' (int, 0/1)
        """

        rng = np.random.default_rng(random_state)

        # -------------------------------
        # 1. Start with a natural baseline
        # -------------------------------
        # Typical organizations have ~5-10% attrition baseline.
        
        df['attrition_prob'] = self.base_prob

        # -------------------------------
        # 2. External factors (Glassdoor, Job Market)
        # -------------------------------
        # Poor Glassdoor rating → more exits
        
        df['attrition_prob'] += self.hook_weights['GlassdoorRating'] * (df['GlassdoorRating'] < 3)

        # Hot job market → more exits
        df['attrition_prob'] += self.hook_weights['JobMarketIndex'] * (df['JobMarketIndex'] > 7)

        # -------------------------------
        # 3. Internal career alignment (NineBox)
        # -------------------------------
        # Low NineBoxScore (performance/potential misalignment) → attrition
        df['attrition_prob'] += self.hook_weights['NineBoxScore'] * (df['NineBoxScore'] <= 3)

        # -------------------------------
        # 4. Social/Team dynamics
        # -------------------------------
        # Weak team cohesion increases attrition
        df['attrition_prob'] += self.interaction_weight * (
            (df['NineBoxScore'] <= 3) & (df['TeamCohesion'] < 0.4)
        )

        # -------------------------------
        # 5. Cross-feature interactions
        # -------------------------------
        # Bad combo: low NineBox & low cohesion → stronger effect
        df['attrition_prob'] += self.interaction_weight * (
            (df['NineBoxScore'] <= 3) & (df['TeamCohesion'] < 0.4)
        )

        # -------------------------------
        # 6. Randomness for realism
        # -------------------------------
        # Add random noise (0 to 5%) so attrition isn't fully explained
        df['attrition_prob'] += rng.uniform(0, self.noise_level, len(df))

        # -------------------------------
        # 7. Clip probabilities to [0, 0.9]
        # -------------------------------
        # Ensure no probability exceeds 90% (nobody has 100% certainty)
        df['attrition_prob'] = df['attrition_prob'].clip(0, self.prob_cap)

        # -------------------------------
        # 8. Sample actual attrition outcomes
        # -------------------------------
        # Use binomial sampling: employee leaves with prob = attrition_prob
        df['Attrition'] = rng.binomial(1, df['attrition_prob'])

        # Safety check: ensure at least 1% positive cases
        min_count = int(self.min_attrition_ratio * len(df))
        if df['Attrition'].sum() < min_count:
            print(f"[Warning] Only {df['Attrition'].sum()} positives found. Forcing {min_count} attrition cases.")
            idx = rng.choice(df.index, size=min_count, replace=False)
            df.loc[idx, 'Attrition'] = 1
        
        # Optional: show final distribution
        final_ratio = df['Attrition'].mean()
        print(f"[Info] Final attrition rate after safeguard: {final_ratio:.2%}")

        return df


# ========================================================================
# Example usage (standalone test)
# ------------------------------------------------------------------------
# if __name__ == "__main__":
#     df = pd.DataFrame({
#         "GlassdoorRating": np.random.randint(1, 6, 1000),
#         "JobMarketIndex": np.random.randint(1, 11, 1000),
#         "NineBoxScore": np.random.randint(1, 10, 1000),
#         "TeamCohesion": np.random.rand(1000)
#     })
#     df = generate_attrition(df)
#     print(df[['GlassdoorRating', 'JobMarketIndex',
#               'NineBoxScore', 'TeamCohesion',
#               'attrition_prob', 'Attrition']].head())
# ========================================================================
