import json
import unittest

from tokenizers import Tokenizer

from research_tokenizer import encode, experiment, morphology_spans, pilot_data


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = experiment(pilot_data())

    def test_saved_artifacts_reproduce_ids_and_unseen_text_exactly(self):
        for name, artifact in self.result["artifacts"].items():
            tokenizer = Tokenizer.from_str(json.dumps(artifact["tokenizer_json"]))
            morphology = artifact["morphology_boundaries"]
            for text in [
                "  먹었어요!\n👩🏽‍🔬\t𠮷",
                "한글\r\n",
                "<|endoftext|>",
                "",
                "가나다XYZ123",
            ]:
                ids = encode(tokenizer, text, morphology)
                self.assertEqual(tokenizer.decode(ids), text, name)
            for row in self.result["rows"]:
                if row["condition"] == name:
                    self.assertEqual(
                        encode(tokenizer, row["text"], morphology), row["ids"]
                    )

    def test_budget_and_summary_are_measured_not_estimated(self):
        for training in self.result["training"].values():
            self.assertEqual(training["utf8_bytes"], len(training["text"].encode()))
            self.assertLessEqual(training["utf8_bytes"], 1000)
        for condition in self.result["conditions"]:
            for lang in ("ko", "en"):
                total = sum(
                    row["token_count"]
                    for row in self.result["rows"]
                    if row["condition"] == condition["name"] and row["language"] == lang
                )
                self.assertEqual(condition[f"{lang}_tokens"], total)

    def test_boundaries_keep_spaces_and_do_not_use_normalized_stems(self):
        text = "  먹었어요.\n한국어를배워요!"
        spans = morphology_spans(text)
        self.assertEqual("".join(spans), text)
        self.assertGreater(len(spans), len(text.split()))

    def test_rejects_leakage_and_invalid_parameters(self):
        for change in [
            {"vocab_size": True},
            {"byte_budget": -1},
            {"test_pairs": []},
            {"train_en": ""},
            {"test_pairs": [{"ko": "나는 아침에 학교에 갑니다.", "en": "Unseen."}]},
        ]:
            with self.assertRaises(ValueError):
                experiment({**pilot_data(), **change})


if __name__ == "__main__":
    unittest.main()
