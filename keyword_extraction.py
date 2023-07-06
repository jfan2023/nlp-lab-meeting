import streamlit
from rake_nltk import Rake
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def extract_keywords(text):
    rake_nltk_model = Rake()
    rake_nltk_model.extract_keywords_from_text(text)
    keyword_extracted = rake_nltk_model.get_ranked_phrases()
    return keyword_extracted


def generate_word_cloud(text, stream):
    # convert list to string and generate
    keyword_extracted = extract_keywords(text)
    unique_string = (" ").join(keyword_extracted)
    wcloud = WordCloud(width=1000, height=500).generate(unique_string)
    plt.figure(figsize=(15, 8))
    plt.imshow(wcloud)
    plt.axis('off')
    stream.pyplot()
