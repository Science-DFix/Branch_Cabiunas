import unittest
import pandas as pd

from src.cnn1d_ae.scoring import compute_comm_fail_mask, describe_alert_context


class TestCommFailMask(unittest.TestCase):
    def test_text_value_lights_mask_for_window(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="1h")
        raw = pd.DataFrame({"a": ["1.0"] * 10, "b": ["2"] * 10}, index=idx)
        raw.iloc[3, 1] = "Comm Fail"
        m = compute_comm_fail_mask(raw, window_hours=2.0)
        self.assertEqual(m.tolist(), [False] * 3 + [True, True] + [False] * 5)

    def test_missing_value_is_not_comm_fail(self):
        idx = pd.date_range("2025-01-01", periods=4, freq="1h")
        raw = pd.DataFrame({"a": ["1", None, "2", "3"]}, index=idx)
        self.assertFalse(compute_comm_fail_mask(raw, window_hours=1.0).any())


class TestAlertContext(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range("2025-01-01", periods=48 * 60, freq="1min")
        op = pd.Series("on", index=self.idx)
        op.iloc[:600] = "off_longo"  # partida às 10:00
        self.op = op
        self.chan = {"vibracao": pd.Series(True, index=self.idx)}
        self.alarms = pd.DataFrame({"t": pd.to_datetime([]), "tag": []})

    def test_post_startup_episode(self):
        eps = pd.DataFrame({"start": [pd.Timestamp("2025-01-01 10:30")], "end": [pd.Timestamp("2025-01-01 12:00")]})
        c = describe_alert_context(eps, self.op, self.chan, self.alarms, ["X"], ["T"])
        self.assertEqual(c.loc[0, "contexto"], "pos-partida")
        self.assertAlmostEqual(c.loc[0, "h_desde_partida"], 1.2, places=1)
        self.assertEqual(c.loc[0, "modelos_acesos"], "vibracao")

    def test_comm_fail_has_priority_and_repetition_counts_previous(self):
        eps = pd.DataFrame({"start": pd.to_datetime(["2025-01-02 00:00", "2025-01-02 20:00"]),
                            "end": pd.to_datetime(["2025-01-02 02:00", "2025-01-02 22:00"])})
        cf = pd.Series(False, index=self.idx)
        cf.loc["2025-01-01 23:00":"2025-01-01 23:30"] = True
        c = describe_alert_context(eps, self.op, self.chan, self.alarms, ["X"], ["T"], comm_fail=cf)
        self.assertEqual(c["contexto"].tolist(), ["falha de comunicacao", "repeticao (1 alerta(s) em 7 d)"])

    def test_temperature_burst(self):
        eps = pd.DataFrame({"start": [pd.Timestamp("2025-01-02 12:00")], "end": [pd.Timestamp("2025-01-02 13:00")]})
        tags = [f"T{i}" for i in range(5)]
        al = pd.DataFrame({"t": [pd.Timestamp("2025-01-02 08:00")] * 5, "tag": tags})
        c = describe_alert_context(eps, self.op, self.chan, al, ["X"], tags)
        self.assertTrue(c.loc[0, "rajada_temperatura_24h"])
        self.assertEqual(c.loc[0, "contexto"], "rajada de alarmes de temperatura")


if __name__ == "__main__":
    unittest.main()
