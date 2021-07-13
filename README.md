### Installation
The code in this repository was created with the standard Anaconda distribution of Python. No additional libraries were utilised. There should be no issues running the code with Python versions 3.*.

### Project Motivation
This project aims to create a web app that will classify messages into disaster reponse requirements. Users will enter a message on the webpage, and this will be classified into 
the type of disaster the message refers to, and also if aid is required.

### File Descriptions
- data/
  - 2 csv files containing messages and classification data used for training the model.
  - 1 .py file for reading the data, cleaning the data and saving the data to a local database

-models/
  - 1 .py file which reads data from previously created database, trains a model on the data and saves the model to a file.

- app/
  - 1 .py file which creates visualisations for web page, and starts the web page using a Flask backend
  - templates/
    - 2 html files 
    
### Running the App
- Clone repository to your local machine
- Navigate to data folder and run process_data.py passing in 3 args, csv containing disaster message, csv containing categories of those messages, filename of the database
'python .\process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'
- Navigate to models folder and run train_classifier.py passing in 2 args, previously created DisasterResponse database, filename to save the model to (must end .pkl)

'python train_classifier.py DisasterResponse.db classifier.pkl'
- Navigate to app folder and run run.py, no args are needed, but it will look for a database in data folder called DisasterResponse.db and a model in models folder called classifier.pkl
 
'python run.py'
- If any problems are encountered ensure that you are running each .py file from it's own directory, not from a parent directory.
 

