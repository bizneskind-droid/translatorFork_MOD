import unittest
from gemini_translator.utils.realign import (
    validate_html_structure,
    wrap_plain_text_to_paragraphs,
)


class TestRealign(unittest.TestCase):
    def test_validate_html_structure_match(self):
        orig = "<p>Chapter 1</p><p>Text paragraph</p>"
        trans = "<p>Глава 1</p><p>Текст абзаца</p>"
        ok, reason, details = validate_html_structure(orig, trans, strict=True)
        self.assertTrue(ok)
        self.assertTrue(details["matched"])
        self.assertEqual(details["src_p"], 2)
        self.assertEqual(details["ru_p"], 2)
        self.assertEqual(details["diff"], 0)

    def test_validate_html_structure_mismatch(self):
        orig = "<p>Chapter 1</p><p>Paragraph 1</p><p>Paragraph 2</p>"
        trans = "<p>Глава 1</p><p>Объединенный текст</p>"
        ok, reason, details = validate_html_structure(orig, trans, strict=True)
        self.assertFalse(ok)
        self.assertFalse(details["matched"])
        self.assertEqual(details["diff"], -1)
        self.assertIn("Расхождение", reason)

    def test_validate_html_structure_missing_p(self):
        orig = "<p>Chapter 1</p><p>Paragraph 1</p>"
        trans = "Глава 1\n\nТекст без тегов"
        ok, reason, details = validate_html_structure(orig, trans)
        self.assertFalse(ok)
        self.assertTrue(details["missing_p_tags"])

    def test_wrap_plain_text_to_paragraphs(self):
        plain = "Глава 1\n\nПервый абзац.\n\nВторой абзац."
        wrapped = wrap_plain_text_to_paragraphs(plain)
        expected = "<p>Глава 1</p>\n<p>Первый абзац.</p>\n<p>Второй абзац.</p>"
        self.assertEqual(wrapped, expected)

    def test_wrap_plain_text_preserves_existing_p(self):
        already_p = "<p>Глава 1</p><p>Текст</p>"
        wrapped = wrap_plain_text_to_paragraphs(already_p)
        self.assertEqual(wrapped, already_p)


if __name__ == "__main__":
    unittest.main()
