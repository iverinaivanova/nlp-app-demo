# nlp-app-demo

======= Automated Linguistic Annotations =======

HOW TO RUN THE APP

PREREQUISITES: 

- Python 3.8 or later version
- install Streamlit https://docs.streamlit.io/library/get-started/installation
- install spaCy (the app was built with version 2.3.2). Note that if you use a higher spaCy version,
it might be incompatible with the current version of the language model. URL: https://spacy.io/usage#source
- download a language model (the app works with the small model for the English language
"en_core_web_sm"). You can download it from terminal: python -m spacy download en_core_web_sm
- install nltk (python3 -m pip install nltk)
- after you install nltk, you can import it in terminal with the command: import nltk 
- then you can download its corpora with the following command: nltk.corpora()
- install keybert (URL: https://pypi.org/project/keybert/)


LOADING THE "nlpapp.py" IN THE IDE

1. Make your way to the following directory: nlp-app-demo > Scripts and then load the "nlpapp.py" in your IDE

2. Go to File > Settings > Project > Python Interpreter and make sure that the project is
connected to the correct version of python 3.8.

3. Add the following packages to your project (if they are not present in the current project). You can add the packages by
pressing the "+" symbol.

Packages:
- streamlit (a python library for building web apps; URL: https://streamlit.io/)
- spaCy 2.3.2 (a python library for NLP; URL: https://spacy.io/usage/linguistic-features)
- nltk (a python library for NLP; URL: https://www.nltk.org/)
- numpy (a python library for working with arrays; URL: https://numpy.org/)
- pandas (a Python-based library for data analysis and manipulation; URL: https://pandas.pydata.org/getting_started.html)
- displacy (a spaCy visualizer of the dependency relations; URL: https://spacy.io/usage/visualizers)
- base64 (to encode binary data that needs to be stored and transferred over media that are designed to deal with ASCII)
- keybert (a keyword extraction technique that uses BERT embeddings; URL: https://pypi.org/project/keybert/)

RUNNING THE APP IN THE LOCAL BROWSER

1. Open Terminal and make sure that you're in the directory in which the "nlpapp.py" file is located.

2. Then type the following command to run the app "streamlit run nlpapp.py"

3. The app should load in your browser window. :-)
