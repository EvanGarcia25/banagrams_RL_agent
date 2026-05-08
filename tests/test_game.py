import unittest

from bananagrams_rl import BananagramsGame, Dictionary, GameConfig


class TestBananagramsGame(unittest.TestCase):
    def setUp(self):
        words = {"A", "I", "AT", "TO", "CAT", "DOG"}
        self.game = BananagramsGame(
            dictionary=Dictionary(words),
            config=GameConfig(grid_size=7, initial_hand_size=0),
            seed=1,
        )
        self.game.hand = ["C", "A", "T", "D", "O", "G"]

    def test_place_and_validate_word(self):
        self.game.place("C", 3, 2)
        self.game.place("A", 3, 3)
        state = self.game.place("T", 3, 4)
        self.assertIn("CAT", state["valid_words"])

    def test_remove_letter(self):
        self.game.place("C", 3, 3)
        self.assertEqual(self.game.grid[3][3], "C")
        self.game.remove(3, 3)
        self.assertEqual(self.game.grid[3][3], "")
        self.assertIn("C", self.game.hand)

    def test_dump_changes_hand_size(self):
        initial_hand_len = len(self.game.hand)
        self.game.dump("C")
        self.assertGreaterEqual(len(self.game.hand), initial_hand_len)


if __name__ == "__main__":
    unittest.main()
