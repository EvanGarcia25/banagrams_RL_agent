import os

FALLBACK_WORDS = {
    "a", "an", "the", "in", "is", "it", "at", "to", "do", "go",
    "be", "or", "on", "up", "no", "so", "as", "by", "we", "my",
    "he", "hi", "me", "of", "if", "am", "ox", "ax", "ex",
    "cat", "bat", "hat", "mat", "rat", "sat", "fat", "pat", "tap",
    "top", "tip", "tin", "tan", "ten", "ton", "net", "pet", "set",
    "get", "let", "bet", "met", "wet", "yet", "bit", "fit", "hit",
    "sit", "wit", "not", "dot", "got", "hot", "lot", "pot", "rot",
    "cut", "but", "gut", "hut", "nut", "put", "run", "fun", "sun",
    "bun", "gun", "nun", "pun", "can", "ban", "fan", "man", "pan",
    "van", "den", "hen", "men", "pen", "ten", "bin", "fin", "kin",
    "pin", "win", "bog", "dog", "fog", "hog", "jog", "log", "cog",
    "bug", "dug", "hug", "jug", "mug", "rug", "tug",
    "cake", "lake", "bake", "fake", "make", "rake", "take", "wake",
    "game", "fame", "name", "same", "came", "tame", "lame",
    "fine", "line", "mine", "nine", "pine", "vine", "wine", "dine",
    "bone", "cone", "tone", "zone", "lone", "done", "none",
    "blue", "clue", "glue", "true", "flue",
    "play", "clay", "day", "say", "way", "bay", "hay", "lay", "may",
    "pay", "ray", "stay", "gray", "pray", "slay",
    "best", "nest", "rest", "test", "vest", "west", "zest",
    "band", "hand", "land", "sand", "and", "end",
    "ring", "sing", "king", "wing", "ding", "ping",
    "back", "hack", "jack", "lack", "pack", "rack", "sack", "tack",
    "deck", "neck", "peck", "beck",
    "bold", "cold", "fold", "gold", "hold", "mold", "old", "sold", "told",
    "base", "case", "face", "lace", "pace", "race", "vase",
    "able", "table", "fable", "cable", "sable",
    "act", "add", "age", "aid", "aim", "air", "all", "apt", "arc", "are",
    "arm", "art", "ask", "ate", "awe", "bad", "bag", "ban", "bar",
    "bed", "big", "box", "boy", "bus", "buy", "cap", "car", "cod",
    "cop", "cow", "cry", "cup", "dab", "dig", "dim", "dip", "dry",
    "dye", "ear", "eat", "egg", "ego", "elf", "elm", "emu", "end",
    "era", "eve", "eye", "fad", "far", "fee", "few", "fib", "fig",
    "fix", "fly", "foe", "fog", "for", "fox", "fry", "fur", "gag",
    "gap", "gas", "gem", "gin", "god", "gum", "gut", "guy", "gym",
    "had", "ham", "has", "haw", "hay", "her", "hew", "hex", "him",
    "his", "hob", "hoe", "hop", "how", "hub", "hue", "hum", "ice",
    "icy", "imp", "ink", "inn", "ion", "ire", "irk", "ivy", "jab",
    "jag", "jam", "jar", "jaw", "jay", "jet", "job", "joe", "jot",
    "joy", "jug", "jut", "keg", "key", "kid", "kin", "kit", "lab",
    "lad", "lag", "lap", "law", "lax", "lea", "led", "leg", "lid",
    "lip", "lit", "lob", "lop", "low", "lug", "mad", "map", "mar",
    "maw", "mob", "mod", "mom", "mop", "mud", "mug", "nab", "nag",
    "nap", "nip", "nob", "nod", "nor", "now", "oak", "odd", "ode",
    "oil", "old", "one", "opt", "orb", "ore", "our", "out", "owe",
    "own", "pad", "pal", "pam", "par", "paw", "pea", "peg", "pew",
    "pie", "pig", "pit", "ply", "pod", "pop", "pry", "pub", "pug",
    "pup", "raw", "red", "ref", "rep", "rev", "rib", "rid", "rig",
    "rip", "rob", "rod", "row", "rub", "rue", "rut", "rye", "sag",
    "sap", "saw", "sea", "see", "sew", "shy", "sin", "sip", "ski",
    "sky", "sly", "sob", "sod", "sop", "sow", "soy", "spa", "spy",
    "sty", "sub", "sue", "sum", "sup", "tab", "tad", "tag", "tar",
    "tax", "tea", "the", "thy", "tic", "tie", "til", "toe", "too",
    "tor", "toy", "try", "tub", "two", "ugh", "ump", "use", "van",
    "vat", "vow", "wag", "war", "was", "wax", "web", "wed", "who",
    "why", "wig", "woe", "wok", "woo", "wow", "yak", "yam", "yap",
    "yew", "you", "zap", "zip", "zoo",
}


class Dictionary:
    def __init__(self, words: set[str]):
        self._words = words

    def is_valid(self, word: str) -> bool:
        return word.lower() in self._words

    @classmethod
    def load(cls) -> "Dictionary":
        for path in ("/usr/share/dict/words", "/usr/dict/words"):
            if os.path.exists(path):
                with open(path) as f:
                    words = {line.strip().lower() for line in f if line.strip().isalpha()}
                return cls(words)
        return cls(FALLBACK_WORDS)
