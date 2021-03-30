# Disaster Response Pipeline Project

As part of my data science studies at Udacity, I built this API from the analysis and construction of a machine learning model carried out on disaster communication data from different sources such as news, social networks or emergency calls, with the purpose to direct the results to the Agencies in charge of immediate attention.
The data set used was provided by Figure Eight, and consists of more than 26,000 messages classified in 36 different categories.
I hope you enjoy this exercise.

##Table of Contents:

1. [Files and Directories](#files)
2. [Installation Guide](#installation)
3. [Running Instructions](#instructions)
4. [Data Analysis and Processes](#data)
5. [Results adn Recommendations](#results)
6. [Screenshots](#screenshots)
7. [Licenses](#licenses)


### Files and Directories: <a name="files"></a>

This repository has the next distribution of files:
* app folder: - where the API html environment is located.
  * templates folder: - html files.
    * go.html
    * master.html
  * run.py - python file to run the html environment or API.
* data folder: - where the datasets are located and processed.
  * disaster_categories.csv - original dataset categorized.
  * disaster_messages.csv - original dataset of messages.
  * process_data.py - ETL pipeline program to process the raw data.
  * DisasterResponse.db - database with ETL data results.
* models folder: - where the ML Pipeline program and results are located.
  * train_classifier.py - the ML pipeline program.
  * DisasterResponseModel.pkl - our resulting trained model - missing for sizing
* notebooks files folder: - a folder with the original commented notebooks.
* screenshots folder: - a folder with the two screenshot showed in this readme.

Note: the final trained model or the pkl file, was not saved or push to this
repository because github don't allow me to save big files like it. It's
necessary to run all the pipelines to create this file and run the final app.


### Installation Guide: <a name="installation"></a>

LIBRARIES REQUIRED TO BE INSTALLED:

* flask
* json
* pandas
* numpy
* sys
* plotly
* re
* joblib
* sqlalchemy
* nltk
* pickle
* sklearn

COPY REPOSITORY:

Now copy this repository to a local directory in your computer


### Running Instructions: <a name="instructions"></a>

For this dataset, or any new one with the same structure, follow the next steps in order to run both, ETL and ML pipelines, and the further API with the resulting model.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/DisasterResponseModel.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Data Analysis and Processes: <a name="data"></a>

####1. ETL PIPELINE:
Our first pipeline makes the next steps in the ETL process, running the
process_data.py file:
1. Download the raw data.
2. Clean the data:
  2.1 Creates de columns for the classified categories.
  2.2 Clean the categories from non binary values.
  2.3 Delete categories with no values.
  2.4 Join all the data.
  2.5 Delete duplicates.
  2.6 Balance the data, oversampling and undersampling some rows.
3. Finally save the data in a database file.

####2. ML PIPELINE:
After step 1, we proceed with the next steps in the ML pipeline, running the
train_classifier.py file:
1. Load the data from the saved database.
2. Build the model with our defined tokenizer and two particular features.
3. Train the model.
4. Evaluate the model and print the results.
5. Save the model as pickle file.

####3. API AND HTML ENVIRONMENT:
Finally running the run.py, we achieve the flask environment that shows a
webpage where we can test the model, predicting the results for any message we
want to create.

### Results and Recommendations: <a name="results"></a>

####RESULTS:
To evaluate our model, we use in this case some different scores: accuracy score, precision, recall and f1-score.
Here are the resulting scores for the final model approached (after some
Gridsearch training):

        Accuracy score: 0.450668896321

                        precision    recall  f1-score   support

               related       0.92      0.99      0.95      4292
               request       0.90      0.77      0.83      1099
                 offer       0.97      0.71      0.82        51
           aid_related       0.86      0.90      0.88      3023
          medical_help       0.98      0.49      0.65       602
      medical_products       1.00      0.55      0.71       423
     search_and_rescue       0.99      0.87      0.93       361
              security       1.00      0.90      0.95       229
              military       0.95      0.81      0.87       345
                 water       0.95      0.64      0.77       473
                  food       0.94      0.76      0.84       748
               shelter       0.97      0.69      0.80       669
              clothing       0.99      0.86      0.92       169
                 money       0.99      0.82      0.89       238
        missing_people       1.00      0.92      0.96       168
              refugees       0.99      0.84      0.91       386
                 death       0.97      0.57      0.72       349
             other_aid       0.93      0.46      0.62       920
infrastructure_related       0.99      0.62      0.76       641
             transport       1.00      0.62      0.76       394
             buildings       0.98      0.54      0.70       382
           electricity       0.99      0.83      0.91       218
                 tools       1.00      0.82      0.90        84
             hospitals       0.98      0.80      0.88       143
                 shops       0.98      0.75      0.85        65
           aid_centers       1.00      0.83      0.91       162
  other_infrastructure       0.98      0.52      0.68       364
       weather_related       0.94      0.85      0.89      1970
                floods       0.99      0.70      0.82       647
                 storm       0.90      0.69      0.78       609
                  fire       0.99      0.83      0.90       141
            earthquake       0.96      0.80      0.87       615
                  cold       0.99      0.80      0.88       231
         other_weather       0.96      0.45      0.61       366
         direct_report       0.88      0.67      0.76      1240
           not_related       0.72      0.20      0.31       492

           avg / total       0.93      0.77      0.83     23309

In this case, with an uneven multi-level-class data, f1-score is the one that give us a better approach to check our model. Accuracy, that was below 0.5 and a bad representation, is not conclusive for this kind of uneven data. In contrast, f1-score is a better approach because it weights an average of precision and recall, meaning that it takes into account false negatives and false positives, in other words, it tries to balance the imbalance. Is not perfect but is a better approach than accuracy.

In this case the f1-score was good, but i am scare that some overfitting was caused by trying to fix the unbalanced data.

#### RECOMMENDATIONS:

This model, with those high scores, could represent that our model is a little bit overfitted, caused in this case for trying to fix the unbalanced data with oversampling and undersampling. There are some parameters in those functions that we can change to improve these results and doing better.

My computer power limited the possibilities to make good Gridsearch analysis, with more parameters, and different algorithms. Time kills the possibilities to practice more. The next step could be to hire some cloud GPU power and test more options.

Finally, about the data, there are more possibilities to improve the tokenizer, implementing more word cleaning and a better lemmatizer. The data is full of regular words with mistakes that put some noise to our training, and that could be cleaned.

### Screenshots: <a name="screenshots"></a>

![screenshot1](/screenshots/screenshot1.png)
![screenshot2](/screenshots/screenshot1.png)

### Licenses: <a name="licenses"></a>

Appen (Figure-8) for the use of their dataset.
MIT License for the use of NLTK Library.
