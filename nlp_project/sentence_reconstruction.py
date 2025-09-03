# -----------------------------
# Synonym Automaton
# -----------------------------
class SentenceAutomaton:
    def __init__(self):
        self.states = {}
        self.start_state = None
        self.accept_states = set()

    def add_state(self, name, is_start=False, is_accept=False):
        self.states[name] = {}
        if is_start:
            self.start_state = name
        if is_accept:
            self.accept_states.add(name)

    def add_transition(self, from_state, to_state, input_word, output_words):
        """
        output_words: list of possible words to emit
        """
        self.states[from_state][input_word] = (to_state, output_words)

    def run(self, words):
        """
        For simplicity, pick the first synonym for each word.
        """
        current_state = self.start_state
        output = []

        for word in words:
            clean_word = word.strip(",.")  # basic punctuation removal
            if clean_word in self.states[current_state]:
                next_state, out_words = self.states[current_state][clean_word]
                output.append(out_words[0])
                current_state = next_state
            else:
                output.append(clean_word)  # unknown words stay the same
        return " ".join(output), current_state in self.accept_states


if __name__ == "__main__":
    # -----------------------------
    # Original sentence
    # -----------------------------
    original = "In fact, I have received the message from the professor, to show me, this, a couple of days ago."

    # Synonym dictionary
    synonyms = {
        "received": ["got", "obtained"],
        "message": ["note", "email"],
        "professor": ["teacher", "lecturer"],
        "show": ["demonstrate", "display"],
        "couple": ["few", "pair"]
    }

    # Tokenize sentence (split on spaces, keep words simple)
    tokens = [w.strip(",.") for w in original.split()]

    # -----------------------------
    # Build automaton
    # -----------------------------
    automaton = SentenceAutomaton()
    prev_state = "S0"
    automaton.add_state(prev_state, is_start=True)

    for i, token in enumerate(tokens, 1):
        state = f"S{i}"
        automaton.add_state(state, is_accept=(i == len(tokens)))
        output_words = synonyms.get(token, [token])
        automaton.add_transition(prev_state, state, token, output_words)
        prev_state = state

    # -----------------------------
    # Run automaton
    # -----------------------------
    rewritten, accepted = automaton.run(tokens)
    print("Original:", original)
    print("Rewritten:", rewritten)
    print("Accepted:", accepted)
