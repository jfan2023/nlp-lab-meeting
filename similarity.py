import spacy


def spacy_string_similarity(original_str, recalled_str):
    if original_str and recalled_str:
        nlp = spacy.load("en_core_web_lg")
        doc1 = nlp(original_str)
        doc2 = nlp(recalled_str)
        similarity_score = round(doc1.similarity(doc2), 2)
    else:
        similarity_score = 0
    return similarity_score
