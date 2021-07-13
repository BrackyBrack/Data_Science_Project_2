import sys
import pandas as pd
import pickle
import nltk
import re
nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(database_filepath):
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM CategorisedMessages', engine)

    X = df.message.values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = list(df.drop(columns=['id', 'message', 'original', 'genre']).columns)
    
    return X, Y, category_names

def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    result = []
    
    text = re.sub(r'[^\w]', ' ', text.lower())
    tokens = word_tokenize(text)
        
    for token in tokens:
        result.append(lemmatizer.lemmatize(token.strip()))
        
    return result


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('model', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    parameters = {
    'model__estimator__max_features':[10, 15, 25, 36],
    'model__estimator__n_estimators':[10, 25]
    }

    cv = GridSearchCV(pipeline, parameters)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        report = classification_report(Y_test[:,i],y_pred[:,i])
        print('{}\n{}\n'.format(category_names[i], report))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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