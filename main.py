# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import streamlit as st
from streamlit_option_menu import option_menu

import tagger
import spacy
import keyword_extraction
from similarity import spacy_string_similarity
import nltk

nltk.download('stopwords')
nltk.download('punkt')
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_data
def load_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    return nlp


@st.cache_data
def load_spacy_model_large():
    nlp = spacy.load("en_core_web_lg")
    return nlp


def show(task_name, col_annotation, html_file=None, text=None, original_text=None, recalled_text=None):
    if task_name == "Keyword Extraction":
        keywords = keyword_extraction.extract_keywords(text=text)
        if len(keywords) >= 5:
            for keyword in keywords[:5]:
                col_annotation.markdown("- " + keyword)
        else:
            for keyword in keywords:
                col_annotation.markdown("- " + keyword)
        keyword_extraction.generate_word_cloud(text=text, stream=col_annotation)
    else:
        with col_annotation.expander("Visualize Annotation", expanded=True):
            for i in html_file:
                st.markdown(i, unsafe_allow_html=True)


def page_design():
    with st.sidebar:
        sidebar_menu = option_menu(
            menu_title="NLP Tasks",
            options=["POS Tagging", "Named Entity Recognition", "Keyword Extraction"]
        )

    st.title("Application of Natural Language Processing in Linguistic Research")
    col1, col2 = st.columns([1,1])
    col1.subheader("[HULC LAB](%s) MEETING " % "https://www.hulclab.eu/")
    col2.subheader("2023-07-06  by Jing Fan")
    st.divider()
    st.header(f"{sidebar_menu}")

    return sidebar_menu


def main():
    selected = page_design()

    if selected == "Similarity Comparison":
        input_options1, input_options2 = st.columns([1, 1])
        text1 = input_options1.text_area(label="Please enter the original texts:", height=80, value="this is a good man")
        text2 = input_options2.text_area(label="Please enter the recalled texts:", height=80, value="the man is nice")
    else:
        if selected == "Keyword Extraction":
            texts = st.text_area(label="Please enter the texts that you want to analyze:", height=80, value="The goal of this project is to develop a new method for the analysis of eye tracking data, which can be used to investigate attention allocation patterns of humans who are presented with dynamic stimuli.")
        else:
            texts = st.text_area(label="Please enter the texts that you want to analyze:", height=80,
                                             value="Peter gave a book to his sister Mary yesterday in Berlin")

    process_button = st.button('Begin Processing')

    if selected == "POS Tagging":
        if process_button:
            with st.spinner('[1/2] Loading SpaCy Model...'):
                nlp = load_spacy_model()
            with st.spinner('[2/2] Processing'):
                html_file = tagger.spacy_analyse_normal_text(nlp, texts, "dep")
                show(selected, st, html_file=html_file)

    elif selected == "Named Entity Recognition":
        if process_button:
            with st.spinner('[1/2] Loading SpaCy Model...'):
                nlp = load_spacy_model()
            with st.spinner('[2/2] Processing'):
                html_file = tagger.spacy_analyse_normal_text(nlp, texts, "ent")
                show(selected, st, html_file=html_file)

    elif selected == "Semantic Role Labeling":
        if process_button:
            with st.spinner('[1/2] Loading AllenNLP Model...'):
                srl_predictor = load_srl_model()
                nlp = load_spacy_model()
            with st.spinner('[2/2] Processing'):

                html_file = srl.get_srl_output(nlp=nlp, srl_predictor=srl_predictor, text=texts)
                show(selected, st, html_file=html_file)

    elif selected == "Coreference Resolution":
        if process_button:
            with st.spinner('[1/2] Loading AllenNLP Model...'):
                coref_model = load_coref_model()
            with st.spinner('[2/2] Processing'):
                html_file = coref.get_coref_output(coref_model=coref_model, text=texts)
                show(selected, st, html_file=html_file)

    elif selected == "Keyword Extraction":
        if process_button:
            show(selected, st, text=texts)
    elif selected == "Similarity Comparison":
        if process_button:
            show(selected, st, original_text=text1, recalled_text=text2)
    else:
        show(selected, st)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
