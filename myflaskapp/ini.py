from flask import Flask
app = Flask(__name__)
from flaskexample import views


def predict():
    scrape=joblib.load('/Users/shwetapai/Desktop/func_scraper.pkl')
    parse=joblib.load('/Users/shwetapai/Desktop/func_parse.pkl')
    cleaning=joblib.load('/Users/shwetapai/Desktop/func_cleaning.pkl')
    normalize=joblib.load('/Users/shwetapai/Desktop/func_normalize.pkl')
    sent_token=joblib.load('/Users/shwetapai/Desktop/func_sent_token.pkl')
    punc_cleaning=joblib.load('/Users/shwetapai/Desktop/func_punc_cleaning.pkl')
    words_token=joblib.load('/Users/shwetapai/Desktop/func_words_token.pkl')
    identify_allcaps=joblib.load('/Users/shwetapai/Desktop/func_identify_allcaps.pkl')
    count_exclamations=joblib.load('/Users/shwetapai/Desktop/func_count_exclamations.pkl')
    count_imp_words=joblib.load('/Users/shwetapai/Desktop/func_count_imp_words.pkl')
    compute_avg_words=joblib.load('/Users/shwetapai/Desktop/func_compute_avg_words.pkl')
    count_paragraphs=joblib.load('/Users/shwetapai/Desktop/func_count_paragraphs.pkl')
    compute_avg_sents_paragraph=joblib.load('/Users/shwetapai/Desktop/func_compute_avg_sents_paragraph.pkl')
    count_images=joblib.load('/Users/shwetapai/Desktop/func_count_images.pkl')
    count_videos=joblib.load('/Users/shwetapai/Desktop/func_count_videos.pkl')
    count_youtube=joblib.load('/Users/shwetapai/Desktop/func_count_youtube.pkl')
    count_hyperlinks=joblib.load('/Users/shwetapai/Desktop/func_count_hyperlinks.pkl')
    count_bold_tags=joblib.load('/Users/shwetapai/Desktop/func_count_bold_tags.pkl')
    emove_punc=joblib.load('/Users/shwetapai/Desktop/func_remove_punc.pkl')

    #helper function1
    def campaign_details(soup):

    # Collect the "About this project" section if available
        try:
            section1 = soup.find(
            'div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists'
        ).get_text(' ')
        except AttributeError:
            section1 = 'section_not_found'

    # Collect the "Risks and challenges" section if available, and remove all
    # unnecessary text
        try:
            section2 = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ) \
            .get_text(' ') \
            .replace('Risks and challenges',' ') \
            .replace('Learn about accountability on Kickstarter',' ')
        except AttributeError:
            section2 = 'section_not_found'

    # Clean both campaign sections
        return {'about': cleaning(section1), 'risks': cleaning(section2)}

    #helper function2

    def preprocess_text(text):
    """Perform text preprocessing such as removing punctuation, lowercasing all
    words, removing stop words and stemming remaining words"""

    # Access stop word dictionary
        stop_words = set(nltk.corpus.stopwords.words('english'))

    # Initialize the Porter stemmer
        porter = nltk.PorterStemmer()

    # Remove punctuation and lowercase each word
        text = remove_punc(text).lower()

    # Remove stop words and stem each word
        return ' '.join(
        porter.stem(term )
        for term in text.split()
        if term not in set(stop_words))

        def feature_extraction(soup, campaign, section):
    """Extract all the features of the text of campaign section"""


    # Compute the number of words in the section
            num_words = len(words_token(campaign[section]))

    # If the section contains no words, assign NaN to 'num_words' to avoid
    # potential division by zero
    if num_words == 0:
        num_words = np.nan

    #If the section isn't available, then return NaN for each meta feature.
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
            count_bold_tags(soup, section) / num_words,          #%of bold tags
            )

      def extract_user_features(hyperlink):
          # Scraping HTML content from hyperlink and  parsing it
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

          #loading model

           clf= joblib.load('/Users/shwetapai/Desktop/trained_classifier.pkl')

           if request.method=='POST':
               comment=request.form['comment']
               data=[comment]

               scraped_html=scrape(data)
               soup = feature_engineering.parse(scraped_html)
               campaign = feature_engineering.get_campaign(soup)


               campaign['about'] = normalize(campaign['about'])
               meta_features = feature_engineering.extract_meta_features(soup,
               campaign,'about')


               scaled_meta_features = scaler.transform([meta_features])

               preprocessed_text = preprocess_text(campaign['about'])
               X_ngrams = vectorizer.transform([preprocessed_text])
               nmf_user=nmf.transform(X_ngrams)
               X_meta_features = sparse.csr_matrix(scaled_meta_features)
               X_full = sparse.hstack([X_meta_features, X_ngrams])




        X_full = sparse.hstack([X_meta_features, X_ngrams])

        # Compute the probability of the project reaching the funding goal
        prob = floor(100 * clf.predict_proba(X_full)[0, 1])

        # Select a custom response based on the probability value and its
        # corresponding custom color
        if prob >= 67:
            blurb = "Your campaign is in good shape!"
            color_choice = "#20C863"
        elif prob >= 33:
            blurb = "There's definitely room for improvement!"
            color_choice = "#FEB308"
        else:
            blurb = "Your campaign needs a lot of work!"
            color_choice = "#CB3335"

        return render_template(
            'output.html',
            the_result=prob,
            project_title=title,
            img_hash=img_link,
            css_hash=css_link,
            link=hyperlink,
            blurb=blurb,
            color_choice=color_choice
        )
    else:
        return render_template('error.html', css_hash=css_link)
