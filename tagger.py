import re
from spacy import displacy


def spacy_analyse_normal_text(nlp_model, text, task):

    html_file = []

    text = re.sub(' +', ' ', text)
    text_split_stored_in_list = text.strip().split('\r\n')
    # remove empty strings from list
    text_split_stored_in_list = list(filter(None, text_split_stored_in_list))

    docs = nlp_model.pipe(text_split_stored_in_list)
    options = {"offset_x": 30}

    for doc in docs:
        html_file.append(displacy.render(doc, style=task, options=options))

    return html_file
