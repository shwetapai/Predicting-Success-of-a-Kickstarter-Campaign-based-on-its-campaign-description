from flask import  Flask,render_template
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import  Flask,render_template
from flask import render_template
from flask import request
from sklearn.externals import joblib
from math import floor
from scipy import sparse
import pandas as pd
import numpy as np
from random import random
import nltk
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures
import requests
from sklearn.decomposition import NMF
from bs4 import BeautifulSoup
from flask import jsonify
import re


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST', 'GET'])
def results():
    if request.method == 'POST':
        hyperlink = request.form['hyperlink']
        probability = predict(hyperlink)
        return jsonify(probability = probability)


def campaign_details(soup):
    try:
        section1 = soup.find(
        'div',
        class_='full-description js-full-description responsive-media ' + \
        'formatted-lists'
        ).get_text(' ')
    except AttributeError:
        section1 = 'section_not_found'

    # Collect the "Risks and challenges" section if available, and remove all unnecessary text
    try:
        section2 = soup.find(
        'div',
        class_='mb3 mb10-sm mb3 js-risks'
        ).get_text(' ').replace('Risks and challenges',' ').replace('Learn about accountability on Kickstarter',' ')
    except AttributeError:
        section2 = 'section_not_found'

        # Clean both campaign sections
    return {'about': cleaning(section1), 'risks': cleaning(section2)}

def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))

    #Initialize the Porter stemmer
    porter = nltk.PorterStemmer()

    # Remove punctuation and lowercase each word
    text = remove_punc(text).lower()

    # Remove stop words and stem each word
    return ' '.join(porter.stem(term) for term in text.split() if term not in stop_words)
            #
def feature_extraction(soup, campaign, section):
    num_words = len(words_token(campaign[section]))
    if num_words == 0:
        num_words = np.nan

    if campaign[section] == 'section_not_found':
        return([np.nan] * 19)
    else:
        return(len(sentences_token(campaign[section])),  #number of the sentence
        num_words,                                # number of words
        len(identify_allcaps(campaign[section])), # number of all_caps
        len(identify_allcaps(campaign[section])) / num_words,  #% of all caps
        count_exclamations(campaign[section]),              #number of exclamations
        count_exclamations(campaign[section]) / num_words,    #% of exclamations
        count_imp_words(campaign[section]),                   #number of buzz words
        count_imp_words(campaign[section]) / num_words,     #% of buzz words
        compute_avg_words(campaign[section]),                #number of avg words
        count_paragraphs(soup, section),                     #number of paragraphs
        compute_avg_sents_paragraph(soup, section),          #number of sentences per paragraph
        compute_avg_words_paragraph(soup, section),          #number of words per paragraph
        count_images(soup, section),                         #number of images
        count_videos(soup, section),                        # number of videos
        count_youtube(soup, section),                       #number of youtube videos
        count_hyperlinks(soup, section),                    #number of hyperlinks
        count_bold_tags(soup, section),                      #number of bold tag
        count_bold_tags(soup, section) / num_words          #%of bold tags
         )

def extract_user_features(hyperlink):

   scraped_html = scrape(hyperlink)
   soup = parse(scraped_html)

   # Collecting section 'campaign' and normalize text
   campaign = campaign_details(soup)
   campaign['about'] = normalize(campaign['about'])

   #  Extract features
   user_features = feature_extraction(soup, campaign, 'about')

   # Preprocessing text in the campaign section
   text_clean = preprocess_text(campaign['about'])

   return user_features, text_clean

def scrape(hyperlink):
    return requests.get(hyperlink)


def parse(scraped_html):
    # Parse the HTML content using an lxml parser
    return BeautifulSoup(scraped_html.text, 'lxml')

def cleaning(text):
    text_cleaned = ' '.join(text.split()).strip()

    # Remove the HTML5 warning for videos
    return text_cleaned.replace("You'll need an HTML5 capable browser to see this content. " + \
        "Play Replay with sound Play with sound 00:00 00:00",' ')

def normalize(text):

    # Tag email addresses with regex
    normalized = re.sub(
        r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
        'emailaddr',
        text
    )

    # Tag hyperlinks with regex
    normalized = re.sub(
        r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
        'httpaddr',
        normalized
    )

    # Tag money amounts with regex
    normalized = re.sub(r'\$\d+(\.\d+)?', 'dollramt', normalized)

    # Tag percentages with regex
    normalized = re.sub(r'\d+(\.\d+)?\%', 'percntg', normalized)

    # Tag phone numbers with regex
    normalized = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr',
        normalized
    )

    # Tag remaining numbers with regex
    return re.sub(r'\d+(\.\d+)?', 'numbr', normalized)

def sentences_token(text):
     # Tokenize the text into sentences
    return nltk.sent_tokenize(text)


def punc_cleaning(text):

    # Remove punctuation with regex
    return re.sub(r'[^\w\d\s]|\_', ' ', text)


def words_token(text):

    # Remove punctuation and then tokenize the text into words
    return [word for word in nltk.word_tokenize(punc_cleaning(text))]


def identify_allcaps(text):

    # Identify all-caps words with regex
    return re.findall(r'\b[A-Z]{2,}', text)


def count_exclamations(text):

    # Count the number of exclamation marks in the text
    return text.count('!')


def count_imp_words(text):
    # Define a set of adjectives used commonly by Apple marketing team
    # according to https://www.youtube.com/watch?v=ZWPqjXYTqYw
    imp_words = frozenset(
        ['revolutionary', 'breakthrough', 'beautiful', 'magical',
        'gorgeous', 'amazing', 'incredible', 'awesome']
    )

    # Count total number of Apple words in the text
    return sum(1 for word in words_token(text) if word in imp_words)



def compute_avg_words(text):

    # Compute the average number of words in each sentence
    return pd.Series(
        [len(words_token(sentence)) for sentence in sentences_token(text)]
    ).mean()



def count_paragraphs(soup, section):

    # Use tree parsing to count the number of paragraphs depending on which
    # section is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('p'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('p'))

def compute_avg_sents_paragraph(soup, section):
    #look at 'about' section
    if section == 'about':
        paragraphs = soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('p')
    elif section == 'risks':
        paragraphs = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('p')

    # Compute the average number of sentences in each paragraph
    return pd.Series(
        [len(sentences_token(paragraph.get_text(' '))) for paragraph in \
         paragraphs]
    ).mean()


def compute_avg_words_paragraph(soup, section):

    # Use tree parsing to identify all paragraphs depending on which section
    # is requested
    if section == 'about':
        paragraphs = soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('p')
    elif section == 'risks':
        paragraphs = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('p')

    # Compute the average number of words in each paragraph
    return pd.Series(
        [len(words_token(paragraph.get_text(' '))) for paragraph in paragraphs]
    ).mean()

def count_images(soup, section):

    # Use tree parsing to identify all image tags depending on which section
    # is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('img'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('img'))

def count_videos(soup, section):

    # Use tree parsing to count all non-YouTube video tags depending on which
    # section is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('div', class_='video-player'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
         ).find_all('div', class_='video-player'))

def count_youtube(soup, section):

    # Initialize total number of YouTube videos
    youtube_count = 0

    # Use tree parsing to identify all iframe tags depending on which section
    # is requested
    if section == 'about':
        iframes = soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
            '-media formatted-lists'
        ).find_all('iframe')
    elif section == 'risks':
        iframes = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('iframe')

    # Since YouTube videos are contained only in iframe tags, determine which
    # iframe tags contain YouTube videos and count them
    for iframe in iframes:
        # Catch any iframes that fail to include a YouTube source link
        try:
            if 'youtube' in iframe.get('src'):
                youtube_count += 1
        except TypeError:
            pass

    return youtube_count


def count_hyperlinks(soup, section):
    """Count the number of hyperlink tags in a campaign section"""
    # Use tree parsing to compute number of hyperlink tags depending on the
    # section requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('a'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('a'))

def count_bold_tags(soup, section):
    """Count the number of bold tags in a campaign section"""

    # Use tree parsing to compute number of bolded text tags depending on which
    # section is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('b'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('b'))

def remove_punc(sentence):
    #cleanr = re.compile('<.*?>')
    #cleantext = re.sub(cleanr, ' ', sentence)
    cleantext = str(sentence).lower()
    #cleaned = re.sub(r'[?|!|\'|"|#|$|%]',r'',cleantext)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleantext)
    cleaned = str(cleaned).lower()

    return cleaned



# Load scaler and vectorizer
scaler = joblib.load('/Users/shwetapai/Desktop/scaler_250.pkl')
vectorizer = joblib.load('/Users/shwetapai/Desktop/vectorizer_250.pkl')
nmf=joblib.load('/Users/shwetapai/Desktop/nmf_250.pkl')
#load polynomial feature
poly_full=joblib.load('/Users/shwetapai/Desktop/poly_250.pkl')

# scrape=joblib.load('/Users/shwetapai/Desktop/func_scraper.pkl')
# parse=joblib.load('/Users/shwetapai/Desktop/func_parse.pkl')
# cleaning=joblib.load('/Users/shwetapai/Desktop/func_cleaning.pkl')
# normalize=joblib.load('/Users/shwetapai/Desktop/func_normalize.pkl')
# sent_token=joblib.load('/Users/shwetapai/Desktop/func_sent_token.pkl')
# punc_cleaning=joblib.load('/Users/shwetapai/Desktop/func_punc_cleaning.pkl')
# words_token=joblib.load('/Users/shwetapai/Desktop/func_words_token.pkl')
# identify_allcaps=joblib.load('/Users/shwetapai/Desktop/func_identify_allcaps.pkl')
# count_exclamations=joblib.load('/Users/shwetapai/Desktop/func_count_exclamations.pkl')
# count_imp_words=joblib.load('/Users/shwetapai/Desktop/func_count_imp_words.pkl')
# compute_avg_words=joblib.load('/Users/shwetapai/Desktop/func_compute_avg_words.pkl')
# count_paragraphs=joblib.load('/Users/shwetapai/Desktop/func_count_paragraphs.pkl')
# compute_avg_sents_paragraph=joblib.load('/Users/shwetapai/Desktop/func_compute_avg_sents_paragraph.pkl')
# count_images=joblib.load('/Users/shwetapai/Desktop/func_count_images.pkl')
# count_videos=joblib.load('/Users/shwetapai/Desktop/func_count_videos.pkl')
# count_youtube=joblib.load('/Users/shwetapai/Desktop/func_count_youtube.pkl')
# count_hyperlinks=joblib.load('/Users/shwetapai/Desktop/func_count_hyperlinks.pkl')
# count_bold_tags=joblib.load('/Users/shwetapai/Desktop/func_count_bold_tags.pkl')
# remove_punc=joblib.load('/Users/shwetapai/Desktop/func_remove_punc.pkl')

#loading model
clf= joblib.load('/Users/shwetapai/Desktop/trained_classifier.pkl')

def predict(kickstarterUrl):
    # Extract title of the project


    meta_features, processed_section = extract_user_features(kickstarterUrl)

    X_ngrams = vectorizer.transform([processed_section])

    no_topics = 20
    nmf_user=nmf.transform(X_ngrams)

    scaled_meta_features = sparse.csr_matrix(scaler.transform([meta_features]))

    feature_vector = sparse.hstack([scaled_meta_features, nmf_user])

    feature_all=feature_vector.todense()

    feature_poly = poly_full.transform(feature_all)

    prob = floor(100 * clf.predict_proba(feature_poly)[0, 1])

    return prob

if __name__=='__main__':
    app.run(debug=True)
