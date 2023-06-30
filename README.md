# NLP Sports Betting

# Project Description
 
A natural language processing classification project designed to predict what programming language is used in Github repos based around sports betting.
 
# Project Goal
 
* Scrape Github for repos with sports betting as a subject matter sorted by number of stars.
* Find the key words that predict what programming language is used.
* Use 2 NLP techniques to construct 3 machine learning classification models to predict what programming language is used in the scraped repos.
* Visualize the results.

# Initial Thoughts
 
The initial hypothesis is that python and R will be the most common languages and will correlate with words such as data, machine learning, and statistics.
 
# The Plan
 
* Acquire data:
    * scrape Github for repos related to sports betting returning a dataframe with the name of the repo, language used, and the readme contents.
    
* Prepare data:
   * Cleaning:
		* encode, decode, and lowercase text
		* remove punctuation and split text into words
        * lemmatize and remove stopwords
 
* Explore data:
   * Answer the following initial questions:
       1. What are the most common words in Python/JavaScript sports betting repos?
       2. Five most common words?
       3. Does number of unique words vary between Python and JavaScript when it comes to sports betting repos?
       4. Are there any words that uniquely identify a language used in sports betting repos?
       
* Modeling:
    * 2 different NLP feature extraction: TF-IDF and TF
    * 3 different models
        * Random forest
        * Logistic regression
        * KNN

* Conclusions:
	* Identify drivers of programming language usage
    * Develop a model that beats baseline

# Data Dictionary

| Feature | Definition (measurement)|
|:--------|:-----------|
|repo| The name of the repository|
|language| The primary programming language used|
|readme_contents| The contents of the readme (english language text)|


# Steps to Reproduce
1) Clone this repo
2) Use https://drive.google.com/file/d/1QG7CtT5o0vKpscB006A_eTOrqtqh_HWz/view?usp=drive_link to download the json file and save it in the appropriate path without changing the file name 
4) Run notebook

Note: If you would like to recreate the Github scraping and API process we used it can be found in the wrangle file in this repo.
 
# Takeaways and Conclusions<br>

* **Exploration** 
    * The 5 most common words for Python and JavaScript repos are odds, http, betting, bet, sport and http, app, user, betting, sport respectively. From this assortment we can surmise that Python is being used largely for analyzing odds while JavaScript is used mainly for app and user interface development. 
    * The five most common words overall are http, app, betting, user, sport.  These words lineup with what one would expect, as many repos are likely to focus on applications that scrape sports betting related websites.
    * The number of unique words vary between Python and JavaScript repos.
    * There are words that uniquely identify program languages, however, they seem to have little to do with unique features of the language 
    (i.e. machine learning, object oriented, etc) and more to do with how the language is used within sport betting community.
        
* **Modeling**
    * Throughout 1st, 2nd, and 3rd iterations KNN outperformed all other models while maintaining a relatively low train to validate delta score. The best KNN model was the bag of words TF-IDF model with a k of 11, beating the baseline of 27% with a 35% accuracy. 




# Recommendation

* In order to more accurately predict program language based on word usage a significantly larger dataset is required. 
* More stop words could be added with additional univariate exploration to remove words commonly used (i.e. http) in technology development that could lead to data noise. 
