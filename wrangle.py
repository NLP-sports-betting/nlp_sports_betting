#wrangle the data
import pandas as pd
import numpy as np

#see the data
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#play with words
import unicodedata
from nltk.corpus import stopwords
import nltk
import re
from pprint import pprint

#split and model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#sql creds

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


#ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""---------------------------------------------------------------------functions to get the data---------------------------------------------------------------"""

"""---------------------------------------------------------------------functions prep the data-----------------------------------------------------------------"""
#takes a row from the dataframe and cleans it

def basic_clean(original):
    '''
    Input: original text or .apply(basic_clean) to entire data frame
    Actions: 
    lowercase everything,
    normalizes everything,
    removes anything that's not a letter, number, whitespace, or single quote
    Output: Cleaned text
    '''
    # lowercase everything
    basic_cleaned = original.lower()
    # normalize unicode characters
    basic_cleaned = unicodedata.normalize('NFKD', basic_cleaned)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    # Replace anything that is not a letter, number, whitespace or a single quote.
    basic_cleaned = re.sub(r'[^a-z0-9\'\s]', '', basic_cleaned)
    
    return basic_cleaned

def tokenize(basic_cleaned):
    '''
    Input: basic_cleaned text string or .apply(tokenize) to entire data frame
    Actions:
    creates the tokenizer
    uses the tokenizer
    Output: clean_tokenize text string
    '''
    #create the tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    #use the tokenizer
    clean_tokenize = tokenize.tokenize(basic_cleaned, return_str=True)
    
    return clean_tokenize

def remove_stopwords(lemma_or_stem, extra_words=[], exclude_words=[]):
    '''
    Input:text string or .apply(remove_stopwords) to entire data frame
    Action: removes standard stop words
    Output: parsed_article
    '''
    # save stopwords
    stopwords_ls = stopwords.words('english')
    # removing any stopwords in exclude list
    stopwords_ls = set(stopwords_ls) - set(exclude_words)
    # adding any stopwords in extra list
    stopwords_ls = stopwords_ls.union(set(extra_words))
    
    # split words in article
    words = lemma_or_stem.split()
    # remove stopwords from list of words
    filtered = [word for word in words if word not in stopwords_ls]
    # join words back together
    parsed_article = ' '.join(filtered)
    
    return parsed_article

def lemmatize(clean_tokenize):
    '''
    Inputs: clean_tokenize
    Actions: creates lemmatizer and applies to each word
    Outputs: clean_tokenize_lemma
    '''
    #create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    #use lemmatize - apply to each word in our string
    lemmas = [wnl.lemmatize(word) for word in clean_tokenize.split()]
    #join words back together
    clean_tokenize_lemma = ' '.join(lemmas)
    
    return clean_tokenize_lemma

def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words

def cleaned(df):
    '''
    This function will clean the df
    drop nulls, replace special characters
    '''
    # drop nulls
    df = df.dropna()
    # replace special characters with space
    df.readme_contents = df.readme_contents.str.replace('[/,_,-,:,"]', ' ', regex=True)
    # replace heavy, check, and mark with nothing
    df.readme_contents = df.readme_contents.str.replace('heavy', '').str.replace('check', '').str.replace('mark', '')
    # create column with clean text. Tokenized, normalized, lemmatized, stop words removed
    df['clean_norm_token'] = df.readme_contents.apply(tokenize).apply(basic_clean).apply(remove_stopwords).apply(lemmatize)
    # replace 124 with nothing. 124 was created by the program removing '|'
    df.clean_norm_token = df.clean_norm_token.str.replace('124', '')
    #in language column replace language with other if it is not in the top 5 languages
    top_5 = df.language.value_counts().head(5).index.tolist()
    df.language = df.language.apply(lambda x: x if x in top_5 else 'other')
    
    return df

"""---------------------------------------------------------------------functions to split the data----------------------------------------------------------------------"""

#splits your data into train, validate, and test sets for cat target var
def split_function_cat_target(df_name, target_varible_column_name):
    train, test = train_test_split(df_name,
                                   random_state=123, #can be whatever you want
                                   test_size=.20,
                                   stratify= df_name[target_varible_column_name])
    
    train, validate = train_test_split(train,
                                   random_state=123,
                                   test_size=.25,
                                   stratify= train[target_varible_column_name])
    return train, validate, test

"""---------------------------------------------------------------------functions to explore the data----------------------------------------------------------------------"""
#function to return the words per language and the frequency of those words
def explore(df,train):
    #create a list of all the words in each language
    python_words = clean(' '.join(train[train.language=='Python']['clean_norm_token']))
    java_script_words = clean(' '.join(train[train.language=='JavaScript']['clean_norm_token']))
    jupyter_notebook_words = clean(' '.join(train[train.language=='Jupyter Notebook']['clean_norm_token']))
    html_words = clean(' '.join(train[train.language=='HTML']['clean_norm_token']))
    r_words = clean(' '.join(train[train.language=='R']['clean_norm_token']))
    other_words = clean(' '.join(train[train.language=='other']['clean_norm_token']))

    all_words = clean(' '.join(df['clean_norm_token']))
    #create a series of the frequency of each word in each language
    python_freq = pd.Series(python_words).value_counts()
    java_script_freq = pd.Series(java_script_words).value_counts()
    jupyter_notebook_freq = pd.Series(jupyter_notebook_words).value_counts()
    html_freq = pd.Series(html_words).value_counts()
    r_freq = pd.Series(r_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()

    all_freq = pd.Series(all_words).value_counts()

    return python_freq, java_script_freq, jupyter_notebook_freq, html_freq, r_freq, other_freq, all_freq, python_words,\
          java_script_words, jupyter_notebook_words, html_words, r_words, other_words, all_words

#makes ngrams depending on the number you put in
def make_ngrams(words, n):
    return pd.Series(nltk.ngrams(words, n)).value_counts().head(20)

#plots the ngrams and single words via wordcloud
def plot_bigrams(language,words):
    word_data = {k[0] + ' ' + k[1]: v for k, v in words.to_dict().items()}
    
    word_img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(word_data)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(word_img)
    plt.axis('off')
    plt.title(f'Top Words for {language}')
    plt.show()

def word_counts_df(python_freq, java_script_freq, jupyter_notebook_freq,
                        html_freq, r_freq, all_freq):
    word_counts = pd.concat([python_freq, java_script_freq, jupyter_notebook_freq,
                            html_freq, r_freq, all_freq], axis=1).fillna(0).astype(int)

    # rename the col names
    word_counts.columns = ['python', 'java_script', 'jupyter_notebook', 'html', 'r', 'all']    
    return word_counts

def top_unique_words(unique_python_words, unique_java_script_words):
    python_unique = [word for word in unique_python_words if word not in unique_java_script_words]
    java_script_unique = [word for word in unique_java_script_words if word not in unique_python_words]
    python_unique = pd.DataFrame(python_unique, columns=['Python Words'])
    java_script_unique = pd.DataFrame(java_script_unique, columns=['Java_script Words'])

    return print(f"{python_unique.head(10)} \n -------------------------------------------- \n{java_script_unique.head(10)}")



def python_wordcloud(python_freq):
    '''
    this funtion will plot a wordcloud for top 40 python words
    '''
    blog_img = WordCloud(background_color='white').generate_from_frequencies(python_freq.head(40))
    plt.figure(figsize=(8, 4))
    plt.imshow(blog_img)
    plt.axis('off')
    plt.show()


def java_script_wordcloud(java_script_freq):
    '''
    this function will plot a wordcloud for top 40 java script words
    '''
    blog_img = WordCloud(background_color='white').generate_from_frequencies(java_script_freq.head(40))
    plt.figure(figsize=(8, 4))
    plt.imshow(blog_img)
    plt.axis('off')
    plt.show()



# Unique word count for python and java script

def unique_words_for_language(python_words, java_script_words):
    '''
    This fucntion will find the number of unique words in python and java script repos
    '''
    
    unique_python_words = list(set(python_words))
    unique_java_script_words = list(set(java_script_words))
    #compare the words in python_words and java_script_words and return unique words from each
    python_unique = [word for word in unique_python_words if word not in unique_java_script_words]
    java_script_unique = [word for word in unique_java_script_words if word not in unique_python_words]
    
    print(f'     Number of unique Python words: {len(python_unique)}')
    print(f'Number of unique Java Script words: {len(java_script_unique)}')






"""---------------------------------------------------------------------functions to set the X, y sets ----------------------------------------------------------------------"""
#sets the X and y variables for train, validate, and test
def X_y_variables(train, validate, test):
    X_train = train.clean_norm_token
    y_train = train.language
    X_validate = validate.clean_norm_token
    y_validate = validate.language
    X_test = test.clean_norm_token
    y_test = test.language

    return X_train, y_train, X_validate, y_validate, X_test, y_test

#sets the X and y variables for train, validate, and test for the bag of words TF
#make my bag of words Term Frequency 
def X_y_variables_bow(X_train, X_validate, X_test):
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_train) 
    X_validate_bow = cv.transform(X_validate)
    X_test_bow = cv.transform(X_test)

    return X_bow, X_validate_bow, X_test_bow

#sets the X and y variables for train, validate, and test for the bag of words TFIDF
#make my bag of words TF-IDF
def X_y_variables_tfidf(X_train, X_validate, X_test):
    tfidf = TfidfVectorizer()
    X_bow = tfidf.fit_transform(X_train) 
    X_validate_bow = tfidf.transform(X_validate)
    X_test_bow = tfidf.transform(X_test)

    return X_bow, X_validate_bow, X_test_bow

#sets the X and y variables for train, validate, and test for the bag of ngrams TFIDF
def X_y_variables_ngrams_tfidf(X_train, X_validate, X_test):
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    X_bow = tfidf.fit_transform(X_train) 
    X_validate_bow = tfidf.transform(X_validate)
    X_test_bow = tfidf.transform(X_test)

    return X_bow, X_validate_bow, X_test_bow