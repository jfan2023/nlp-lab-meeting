from allennlp.predictors.predictor import Predictor
from spacy import displacy


def get_coref_output(coref_model, text):
    coref_annotations = coref_model.predict(text)
    formated_spans = []
    for index, cluster in enumerate(coref_annotations["clusters"]):
        # Catch cases where annotations were running into problems
        if cluster == []:
            continue
        for span in cluster:
            formated_spans.append({"start_token": span[0], "end_token": span[1]+1, "label": str(index)})
    span_input = {"text": text, "spans": formated_spans, "tokens": coref_annotations["document"]}

    options = {"spans_key": "custom", "colors": {"0": "yellow", "1": "orange", "2": "#3dff74", "3": "#cfc5ff"}}

    html_file = displacy.render(span_input, style="span", manual=True, options=options)

    return html_file
