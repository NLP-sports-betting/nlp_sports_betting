# NLP Sports Betting

# Project Description
 
A natural language processing classification project designed to predict what programming language is used in Github repos based around sports betting.
 
# Project Goal
 
* Scrape Github for repos with sports betting as a subject matter sorted by number of stars.
* Find the key words that predict what programming language is used.
* Use 2 NLP techniques to construct 3 machine learning classification models to predict what programming language is used in the scraped repos.
* Visualize the results.

# Initial Thoughts
 
The initial hypothesis is that .
 
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
       1. What is the top language used?
       2. ?
       3. Does number of stars correlate with the language used?
       4. ?
       
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
2) Go to https:// and download the CSV and save it in the appropriate path without changing the file name 
4) Run notebook
 
# Takeaways and Conclusions<br>

* **Exploration** 
        
* **Modeling**




# Recommendation
