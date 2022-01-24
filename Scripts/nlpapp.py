import streamlit as st
import os
# NLP Packages
# from textblob import TextBlob
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
sentiment_analysis= SentimentIntensityAnalyzer()
import numpy
# from nltk.corpus import treebank
# from rake_nltk import Rake
# r = Rake()
from keybert import KeyBERT
bert = KeyBERT()
import pandas as pd
from spacy import displacy
import base64
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# vader_analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Tokenizer & lemmatizer and POS tagger with Spacy
@st.cache
def tokenizer_lemmatizer(my_text):
    doc = nlp(my_text)
    nlp_data = [(token.text, token.lemma_) for token in doc]
    return nlp_data

# POS tagger with Spacy
def pos_tagger(my_text):
    doc = nlp(my_text)
    nlp_data = [(token.text, token.pos_) for token in doc]
    return nlp_data

# NP Chunker with SpaCy
def np_chunker(my_text):
    doc = nlp(my_text)
    nps = [np.text for np in doc.noun_chunks]
    return nps

# Dependency Relations
def dep_parser(my_text):
    doc = nlp(my_text)
    nlp_data = [(token.text, token.dep_, token.head.text, token.head.pos_) for token in doc]
    return nlp_data

# Dependency Visualizer
def dep_visualizer(my_text):
    doc = nlp(my_text)
    sentences = list(doc.sents)
    design_options = {"compact": False, "bg": "#ffffff",
               "color": "#000000", "font": "Source Source Sans Pro"}
    nlp_data = displacy.render(sentences, style='dep', options=design_options)
    return nlp_data

# Sentiment analyzer
def sentiment_analyzer(my_text):
    score = sentiment_analysis.polarity_scores(my_text)
    return score

# Lexical word extractor (nouns, verbs, adjectives, adverbs)
def lexword_extractor(my_text):
    doc = nlp(my_text)
    nouns = []
    verbs = []
    adjs = []
    advs = []
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.append(token.text)
        if token.pos_ == "VERB":
            verbs.append(token.text)
        if token.pos_ == "ADJ":
            adjs.append(token.text)
        if token.pos_ == "ADV":
            advs.append(token.text)
    lex_words = {}
    lex_words['nouns'] = nouns
    lex_words['verbs'] = verbs
    lex_words['adjectives'] = adjs
    lex_words['adverbs'] = advs
    return lex_words

# Keywords
def keyword_extractor(my_text):
        # keywords = bert.extract_keywords(my_text, keyphrase_ngram_range=(3, 5), stop_words="english", top_n=5)
        keywords = bert.extract_keywords(my_text, stop_words="english", top_n=10)
        results = []
        for scored_keywords in keywords:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # Downloading the table as csv
def download_table_as_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="table.csv" target="_blank">Download the table as csv</a>'
    return href


# The main() function
def main():
    st.header("Automated Linguistic Annotations")
    input_text = st.text_area("Enter Text")
    st.sidebar.header("Annotator(s)")
    annotators = st.sidebar.multiselect("Select annotators", ["tokens & lemmas", "pos", "dependency rel",
                                                             "dep visualizer", "sentiment", "NP", "lexical words",
                                                              "keywords"])
    button1 = st.sidebar.button("Annotate")
    if button1:
        if "tokens & lemmas" in annotators:
            tokens = tokenizer_lemmatizer(input_text)
            # st.write(tokens)
            df = pd.DataFrame(tokens, columns=['token', 'lemma_'])
            st.subheader("Tokens and Lemmas (lemma)")
            st.write(df)
            download_table = download_table_as_csv(df)
            st.write(download_table, unsafe_allow_html=True)

        if "pos" in annotators:
            pos = pos_tagger(input_text)
            # st.write(pos)
            df = pd.DataFrame(pos, columns=['token', 'pos'])
            st.subheader("Tokens and Parts-of-Speech (pos)")
            st.write(df)
            reference_pos = 'Reference to the [POS tags] (https://universaldependencies.org/u/pos/)'
            st.write(reference_pos)
            download_table = download_table_as_csv(df)
            st.write(download_table, unsafe_allow_html=True)

        if "dependency rel" in annotators:
            dep = dep_parser(input_text)
            # st.write(dep)
            df = pd.DataFrame(dep, columns=['token', 'dep', 'head', 'head pos'])
            st.subheader("Tokens, Types of syntactic relations (dep), Head word, Part-of-speech of the head (head pos)")
            st.write(df)
            reference_pos = 'Reference to the [POS tags] (https://universaldependencies.org/u/pos/)'
            st.write(reference_pos)
            reference_dep = 'Reference to the [dependency relations tags] (https://universaldependencies.org/u/dep/)'
            st.write(reference_dep)
            download_table = download_table_as_csv(df)
            st.write(download_table, unsafe_allow_html=True)

        if "dep visualizer" in annotators:
            dep_rel = dep_visualizer(input_text)
            st.subheader("Visualizing the dependency relations between the tokens")
            st.image(dep_rel, width=10)


        if "sentiment" in annotators:
            emotions = sentiment_analyzer(input_text)
            st.write(emotions)
            df_emotions = pd.DataFrame([emotions])
            # emo = pd.Series(emotions)
            # emo2 = emo.to_frame()
            st.subheader("Sentiment Analysis")
            # st.write(df_emotions)
            st.write('''
            - If the compound score is > 0, the sentiment is positive. 
            - If the compound score is < 0, the sentiment is negative. 
            '''
            )
            st.write(df_emotions)
            download_table = download_table_as_csv(df_emotions)
            st.write(download_table, unsafe_allow_html=True)

        if "NP" in annotators:
            np_chunks = np_chunker(input_text)
            st.subheader("Noun Phrases")
            df_nps = pd.DataFrame(np_chunks, columns=["Noun Phrases"])
            st.write(df_nps)
            download_table = download_table_as_csv(df_nps)
            st.write(download_table, unsafe_allow_html=True)

        if "lexical words" in annotators:
            words = lexword_extractor(input_text)
            st.subheader("Nouns, verbs, adjectives, adverbs")
            lex_words = pd.DataFrame.from_dict(words, orient="index")
            st.write(lex_words)
            download_table = download_table_as_csv(lex_words)
            st.write(download_table, unsafe_allow_html=True)

        if "keywords" in annotators:
            keywords_extract = keyword_extractor(input_text)
            st.subheader("Keywords")
            df_keywords = pd.DataFrame(keywords_extract, columns = ['Keywords'])
            st.write(df_keywords)
            download_table = download_table_as_csv(df_keywords)
            st.write(download_table, unsafe_allow_html=True)

    st.sidebar.markdown('''
    This Natural Language Processing (NLP) App: 
    - tokenizes the input sentence/text
    - assigns parts of speech to the tokens
    - extracts the lemmas
    - extracts the Noun Phrases (NPs)
    - extracts the lexical words from the input sentence/text
    - parses the dependency relations
    - visualizes the dependency relations
    - analyzes the sentiment of the text
    - extracts the keywords of the input text
    '''
    )
if __name__ == '__main__':
    main()