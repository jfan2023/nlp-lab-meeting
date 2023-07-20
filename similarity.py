import spacy


def spacy_string_similarity(nlp_model, original_str, recalled_str):
    if original_str and recalled_str:
        doc1 = nlp_model(original_str)
        doc2 = nlp_model(recalled_str)
        similarity_score = round(doc1.similarity(doc2), 2)
    else:
        similarity_score = 0
    return similarity_score
