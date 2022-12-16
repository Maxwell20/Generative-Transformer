import unittest

class TestGenerateText(unittest.TestCase):
    def test_generated_text_length(self):
        # Test that the generated text has the expected length
        generated_text = generate_text(model, 'The', 100)
        self.assertEqual(len(generated_text), 100)

    def test_generated_text_starts_with_seed(self):
        # Test that the generated text starts with the seed string
        generated_text = generate_text(model, 'The', 100)
        self.assertTrue(generated_text.startswith('The'))

    def test_generated_text_contains_only_allowed_chars(self):
        # Test that the generated text contains only characters from the original text
        generated_text = generate_text(model, 'The', 100)
        for c in generated_text:
            self.assertIn(c, vocab)

if __name__ == '__main__':
    unittest.main()
