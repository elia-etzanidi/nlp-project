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
    
    def process_sentence(self, original, synonyms):
        # Reset states
        self.states = {}
        self.start_state = None
        self.accept_states = set()

        # Tokenize
        tokens = [w.strip(",.") for w in original.split()]

        # Build automaton
        prev_state = "S0"
        self.add_state(prev_state, is_start=True)
        for i, token in enumerate(tokens, 1):
            state = f"S{i}"
            self.add_state(state, is_accept=(i == len(tokens)))
            output_words = synonyms.get(token, [token])
            self.add_transition(prev_state, state, token, output_words)
            prev_state = state

        # Run
        rewritten, accepted = self.run(tokens)

        print("Original:", original)
        print("Rewritten:", rewritten)
        print("Accepted:", accepted)
        print("\n","-" * 50)
        return rewritten, accepted


if __name__ == "__main__":
    automaton = SentenceAutomaton()

    original_1 = "In fact, I have received the message from the professor, to show me, this, a couple of days ago."
    synonyms_1 = {
        "received": ["got", "obtained"],
        "message": ["note", "email"],
        "professor": ["teacher", "lecturer"],
        "show": ["demonstrate", "display"],
        "couple": ["few", "pair"]
    }
    print("\nFirst Sentence Processing:\n")
    automaton.process_sentence(original_1, synonyms_1)

    original_2 = "Because I didn’t see that part final yet, or maybe I missed, I apologize if so."
    synonyms_2 = {
        "didn’t": ["did not"],
        "see": ["notice", "view"],
        "final": ["finished", "complete"],
        "yet": ["so far", "up to now"],
        "maybe": ["perhaps"],
        "missed": ["overlooked"],
        "apologize": ["am sorry"]
    }
    print("\nSecond Sentence Processing:\n")
    automaton.process_sentence(original_2, synonyms_2)
