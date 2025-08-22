import re
from collections import defaultdict

def parse_conllu(file_path):
    """Parses a CoNLL-U file into a list of sentences, where each sentence is a list of word dictionaries."""
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":  # Sentence delimiter
                if 0 < len(sentence) <= 8:  # Discard sentences longer than 8 words
                    sentences.append(sentence)
                sentence = []
            elif not line.startswith("#"):  # Ignore metadata
                columns = line.strip().split("\t")
                if len(columns) >= 4:
                    sentence.append({
                        "id": columns[0],
                        "form": columns[1],
                        "lemma": columns[2],
                        "upos": columns[3]
                    })
    if 0 < len(sentence) <= 8:  # Ensure last sentence is included if valid
        sentences.append(sentence)
    return sentences

def find_matching_sentences(conllu_file):
    sentences = parse_conllu(conllu_file)
    lemma_dict = defaultdict(lambda: {"start": None, "middle": None, "title": None})

    for sentence in sentences:
        if not sentence:
            continue

        first_word = sentence[0]["form"]
        first_lemma = sentence[0]["lemma"]

        # Check if it's a title (all words capitalized)
        is_title = all(word["form"].isupper() for word in sentence)  

        for i, word in enumerate(sentence):
            lemma = word["lemma"]
            form = word["form"]

            # Start-of-sentence case: first letter capitalized, not all uppercase
            if i == 0 and form[0].isupper() and not form.isupper() and len(lemma) > 1:
                lemma_dict[lemma]["start"] = sentence  

            # Middle of the sentence case: lowercase form
            elif i > 0 and form.islower():
                lemma_dict[lemma]["middle"] = sentence  

            # Title case: full uppercase lemma
            if is_title and form == lemma.upper():
                lemma_dict[lemma]["title"] = sentence  

    # Find lemmas that match all three criteria
    matching = {lemma: sents for lemma, sents in lemma_dict.items() if all(sents.values())}

    return matching

# Example usage
conllu_file = "cs_pdt-ud-train.conllu"
matching_sentences = find_matching_sentences(conllu_file)

# Print results
for lemma, sents in matching_sentences.items():
    print(f"\nLemma: {lemma}")
    for category, sentence in sents.items():
        print(f"  {category.capitalize()} Sentence: {' '.join(w['form'] for w in sentence)}")
