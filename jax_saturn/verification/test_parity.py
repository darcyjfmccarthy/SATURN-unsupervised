"""Guard against accepting incomplete, incomparable or degraded metric tables."""

import unittest

import pandas as pd

from jax_saturn.verification.parity import compare_metrics, TRIALS


class ParityReportTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({"trial": TRIALS, "fixed_triplet_count": [100] * 4,
                                       "fixed_triplet_margin_loss": [.1] * 4,
                                       "label_same_neighbor_fraction": [.8] * 4,
                                       "species_mixing_fraction": [.2] * 4})

    def test_directional_budgets_and_trial_alignment(self):
        candidate = self.reference.iloc[::-1].copy()
        candidate["fixed_triplet_margin_loss"] = .12
        candidate["label_same_neighbor_fraction"] = .75
        candidate["species_mixing_fraction"] = .17
        self.assertTrue(compare_metrics(self.reference, candidate)["passes"].all())
        candidate.loc[candidate["trial"] == "ot", "fixed_triplet_margin_loss"] = .121
        failed = compare_metrics(self.reference, candidate).query("passes == False")
        self.assertEqual(failed["trial"].tolist(), ["ot"])
        improved = self.reference.copy()
        improved["fixed_triplet_margin_loss"] = 0
        improved["label_same_neighbor_fraction"] = 1
        improved["species_mixing_fraction"] = 1
        self.assertTrue(compare_metrics(self.reference, improved)["passes"].all())

    def test_incomplete_invalid_and_different_triplets_are_rejected(self):
        bad_tables = [self.reference.iloc[:3], pd.concat([self.reference.iloc[:3], self.reference.iloc[:1]])]
        for field, value in (("fixed_triplet_count", 101), ("fixed_triplet_count", 0),
                             ("fixed_triplet_count", 100.5), ("fixed_triplet_margin_loss", float("nan")),
                             ("species_mixing_fraction", 1.1), ("label_same_neighbor_fraction", -.1)):
            bad = self.reference.astype({"fixed_triplet_count": float}).copy()
            bad.loc[0, field] = value
            bad_tables.append(bad)
        for bad in bad_tables:
            with self.subTest(table=bad.to_dict()), self.assertRaises(ValueError):
                compare_metrics(self.reference, bad)
        with self.assertRaises(ValueError):
            compare_metrics(self.reference, self.reference.drop(columns=["fixed_triplet_count"]))


if __name__ == "__main__":
    unittest.main()
