import unittest
import pandas as pd
from src.preprocessing import add_traffic_features, clean_column_names, normalize_target_values

class TestPreprocessing(unittest.TestCase):
    def test_seven_engineered_features_are_created(self):
        frame = pd.DataFrame({
            "sbytes": [100, 50], "dbytes": [20, 10],
            "spkts": [10, 5], "dpkts": [2, 1], "dur": [2.0, 1.0]
        })
        result = add_traffic_features(frame)
        expected = {
            "derived_total_bytes", "derived_src_dst_byte_log_ratio",
            "derived_total_packets", "derived_src_dst_packet_log_ratio",
            "derived_mean_bytes_per_packet", "derived_bytes_per_duration",
            "derived_packets_per_duration"
        }
        self.assertTrue(expected.issubset(result.columns))

    def test_feature_engineering_does_not_modify_input(self):
        frame = pd.DataFrame({"sbytes": [10], "dbytes": [5], "spkts": [2], "dpkts": [1], "dur": [1.0]})
        original_columns = list(frame.columns)
        add_traffic_features(frame)
        self.assertEqual(list(frame.columns), original_columns)

    def test_zero_packet_count_is_guarded(self):
        frame = pd.DataFrame({"sbytes": [10], "dbytes": [5], "spkts": [0], "dpkts": [0], "dur": [1.0]})
        result = add_traffic_features(frame)
        self.assertTrue(pd.notna(result.loc[0, "derived_mean_bytes_per_packet"]))

    def test_column_names_are_trimmed(self):
        frame = pd.DataFrame([[1, 2]], columns=[" sbytes ", " dbytes "])
        result = clean_column_names(frame)
        self.assertEqual(list(result.columns), ["sbytes", "dbytes"])

    def test_duplicate_names_after_trim_are_rejected(self):
        frame = pd.DataFrame([[1, 2]], columns=["sbytes", " sbytes "])
        with self.assertRaises(ValueError):
            clean_column_names(frame)

    def test_target_labels_are_normalized(self):
        labels = normalize_target_values(pd.Series([" normal ", "DOS", "Worms"]))
        self.assertEqual(labels.tolist(), ["Normal", "DoS", "Worms"])

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_target_values(pd.Series(["unknown"]))

if __name__ == "__main__":
    unittest.main()
