import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import load_env_file
from dictionary import Dictionaries


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / ".env"

    def test_keys_are_loaded_for_dictionary_client(self):
        self.path.write_text(
            '# comment\nSTDICT_API_KEY="test-standard" # comment\n'
            "export OPENDICT_API_KEY='test-open'\nEMPTY=\n",
            encoding="utf-8-sig",
        )
        with patch.dict(os.environ, {}, clear=True):
            load_env_file(self.path)
            self.assertEqual(
                Dictionaries().keys,
                {"stdict": "test-standard", "opendict": "test-open"},
            )
            self.assertEqual(os.environ["EMPTY"], "")

    def test_existing_environment_has_priority(self):
        self.path.write_text('STDICT_API_KEY="file-key"', encoding="utf-8")
        with patch.dict(os.environ, {"STDICT_API_KEY": "terminal-key"}, clear=True):
            load_env_file(self.path)
            self.assertEqual(os.environ["STDICT_API_KEY"], "terminal-key")

    def test_missing_file_is_optional(self):
        load_env_file(self.path)

    def test_invalid_line_does_not_expose_secret(self):
        for line in [
            'STDICT_API_KEY="secret',
            "STDICT_API_KEY=secret extra",
            "not-an-assignment",
        ]:
            self.path.write_text(line, encoding="utf-8")
            with self.assertRaises(ValueError) as error:
                load_env_file(self.path)
            self.assertIn("1번째 줄", str(error.exception))
            self.assertNotIn("secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
