import warnings
from typing import List, Tuple


import allennlp_models.tagging
import spacy
from spacy import displacy

from custom_datatype import CustomSpan


def _convert_tags_to_spans(
    tags: List[str], offset: int
) -> List[Tuple[CustomSpan, str]]:
    """
    Method that converts a BIO tagging sequence (e.g., ["O", "B-ARG1", "B-ARG2", "B-V", "O", "O"])
    into a sequence of spans with associated labels (e.g., "[([1, 2], "ARG1"), ([2, 3], "ARG2"), ([3, 4], "V")].
    This can be offset by an integer amount, which is required for multi-sentence matching to work.
    """
    all_spans = []
    curr_span = []
    curr_label = ""
    for idx, tag in enumerate(tags):
        if tag == "O" or tag.startswith("B-"):
            # We have some previous span. Finish it off with the index and then return
            if curr_span != []:
                curr_span.append(idx)
                all_spans.append(
                    (
                        CustomSpan(
                            start=curr_span[0] + offset, end=curr_span[1] + offset
                        ),
                        curr_label,
                    )
                )
                if tag == "O":
                    curr_span = []
                    curr_label = ""
                else:
                    curr_span = [idx]
                    curr_label = tag[2:]
            # No entry in the span, define a starting position
            else:
                if tag.startswith("B-"):
                    curr_label = tag[2:]
                    curr_span.append(idx)

        # For intermediate tags, simply continue, since we're only interested in boundaries
        if tag.startswith("I-"):
            continue

    # Finish any last elements
    if curr_span != []:
        # idx is guaranteed to exist, since otherwise curr_span would be empty
        curr_span.append(idx + 1)
        all_spans.append(
            (
                CustomSpan(start=curr_span[0] + offset, end=curr_span[1] + offset),
                curr_label,
            )
        )

    return all_spans


def get_srl_output(nlp, srl_predictor, text):

    doc = nlp(text)

    # SRL extraction works only at sentence-level.
    srl = []
    # Also specifically catch invalid sentences, usually due to tables, which will be ignored.
    for sent in doc.sents:
        # Catching errors likely due to incorrect token normalization
        try:
            result = srl_predictor.predict(sent.text)
        except RuntimeError as e:
            result = []
            warnings.warn(
                f"Processing sentence caused an error in the SRL model! You might want to investigate!\n"
                f"Error message: '{e}'\n"
                f"Responsible sentence '{sent.text}'"
            )
        srl.append(result)

    rendering_formats = []
    for srl_annotations, sentence in zip(srl, doc.sents):
        # Catch cases where annotations were running into problems
        if srl_annotations == []:
            continue
        for annotation in srl_annotations["verbs"]:

            spans = _convert_tags_to_spans(
                annotation["tags"], offset=sentence.start
            )
            formated_spans = []
            for span in spans:
                formated_spans.append({"start_token": span[0].start, "end_token": span[0].end, "label": span[1]})
            span_input = {"text": text, "spans": formated_spans, "tokens": srl_annotations["words"]}
            rendering_formats.append(span_input)

    html_files = []
    options = {"spans_key": "custom", "colors": {"ARG0": "yellow", "ARG1": "orange", "ARG2": "#3dff74", "V": "#cfc5ff"}}
    for rendering in rendering_formats:
        html_file = displacy.render(rendering, style="span", manual=True, options=options)
        html_files.append("<br>")
        html_files.append(html_file)

    return html_files
