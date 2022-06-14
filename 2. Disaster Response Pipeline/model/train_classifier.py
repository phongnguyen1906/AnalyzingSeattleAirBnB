import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import pickle

nltk.download(['omw-1.4', 'punkt', 'wordnet'])

def load_data(database_filepath):
    '''load data from database file 
    Input:
    database_filepath: database file
    Return:
    X: dataframe contains feature
    y: dataframe contains label
    column_names: list of categorical labels
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('message_table', con=engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    column_names = y.columns
    return X, y, column_names



def tokenize(text):
    ''' tokenize and lemmatize text
    Input: text string
    Return: list of lemmatized tokens
    '''
    #tokenize text
    tokens = word_tokenize(text)
    #lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmed_tokens = [lemmatizer.lemmatize(t).lower().strip() for t in tokens]
    return lemmed_tokens


def build_model():
    '''ininiate classifier and transformer then build pipeline with GridSearchCV'''
    
    #initiate pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('rf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    #parameters for GridSearch
    parameters = {
        'rf__estimator__n_estimators': [50, 100, 150],
        'rf__estimator__max_depth': [6, 8, 10]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv    

def evaluate_model(model, X_test, y_test, column_names):
    ''' predict and evaluate model
    Input: 
    X_test : feature dataframe for predicting
    y_test : true labels
    column_names : list of labels for multi output classifier
    Output: 
    Model score (f1, precision, recall)
    '''
    #predict and print result:
    y_pred = model.predict(X_test)
    for i, col in enumerate(column_names):
        print(i, col)
        print(classification_report(y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    '''save model into pickle file to run later
    Input: 
    model: fit and trained model
    model_filepath : name of model to save as
    Return:
    model file (*.pkl) in directory
    '''
    #save as pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, column_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, column_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
