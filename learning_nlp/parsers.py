import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

def get_agreement_type(dat, doc_id):
    """
    Extracts and returns phrases formed by a target word (AGREEMENT or CONTRACT) and all its relevant dependents,
    including dependents of dependents, in the exact order they appear in the text.
    Excludes any dependent word that is 'Exhibit' (case insensitive).

    Parameters:
    - dat (dict): The dataset containing text documents.
    - doc_id (int): The document ID to access specific text data in 'dat'.

    Returns:
    - List of phrases formed by the target word, its dependents, and connected words, in the order they appear.
    """
    # Get the context text for the specified document ID
    txt = dat['data'][doc_id]['paragraphs'][0]['context']
    doc = nlp(txt)  # Process the text with Stanza
    results = []

    target_words = ["AGREEMENT", "CONTRACT"]
    # Include 'amod' in the list of dependency relations
    relevant_deprels = ["compound", "cc", "amod"]

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text == "Exhibit":  # Exclude 'Exhibit' as the target word
                continue
            if word.text in target_words:  # Exact case match with 'AGREEMENT' or 'CONTRACT'
                phrase_parts = []

                # Recursive function to collect dependents
                def collect_dependents_recursive(current_word):
                    collected = [(current_word.text, current_word.id)]
                    for child in sentence.words:
                        if (child.head == current_word.id and 
                            child.deprel in relevant_deprels and 
                            child.text.lower() != "exhibit"):
                            collected.extend(collect_dependents_recursive(child))
                    return collected

                # Start collecting from the target word
                phrase_parts.extend(collect_dependents_recursive(word))

                # If the target word has a 'conj' deprel, include its head and its dependents
                if word.deprel == "conj" and word.head > 0:
                    head_word = sentence.words[word.head - 1]
                    if head_word.text.lower() != "exhibit":
                        phrase_parts.extend(collect_dependents_recursive(head_word))

                # Remove duplicates by converting to a set and back to a list
                phrase_parts = list(set(phrase_parts))

                # Sort phrase parts by their IDs to maintain the original order
                phrase_parts = sorted(phrase_parts, key=lambda x: x[1])

                # Join the words in order and add to results
                ordered_phrase = " ".join([part[0] for part in phrase_parts])
                results.append(ordered_phrase)

    return results

# Usage example
# Assuming 'dat' is your dataset and 'nlp' is the initialized Stanza pipeline
#phrases = get_agreement_type(dat, doc_id=5)
#print(phrases)
