class TextReconstructor:
    def __init__(self, file1, file2):
        self.text1 = self._read_file(file1)
        self.text2 = self._read_file(file2)
        self.sentences1 = self._split_into_sentences(self.text1)
        self.sentences2 = self._split_into_sentences(self.text2)

    def _read_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    def _split_into_sentences(self, text):
        import re
        # Split by ., !, ? followed by space or end of text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove empty strings and strip extra spaces
        # is this needed ???
        # return [s.strip() for s in sentences if s.strip()]

    def select_sentence(self, index, text_number=1):
        if text_number == 1:
            if 1 <= index <= len(self.sentences1):
                return self.sentences1[index - 1]
        elif text_number == 2:
            if 1 <= index <= len(self.sentences2):
                return self.sentences2[index - 1]
        raise IndexError("Sentence index out of range.")

    def reconstruct_sen(self, index1, index2):
        sentence1 = self.select_sentence(index1, 1)
        sentence2 = self.select_sentence(index2, 2)
        return f"{sentence1} {sentence2}"
    
if __name__ == "__main__":
    recon = TextReconstructor("texts/text1.txt", "texts/text2.txt")
    result = recon.reconstruct_sen(1, 2)
    print(result)