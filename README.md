# Disaster Response Pipeline Project
### Project Description:

This project is made for the purpose of classifying the disaster messages into categories in real time through the interaction with web app. The data set contains encoded messages from 36 different categories. 

### Installations

To clone this repository:
```
$ git clone https://github.com/msyd1/Disaster_Response.git
```
To install dependencies:
```
$ pip install -r requirements.txt
```
Or install requred libraries:
- pandas
 - json
 - sklearn
 -  nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

### Instructions:

**ETL pipeline**:
This script cleans provided data set and saves it into a database 
Run the following commands in the *data*  directory: 
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
**ML pipeline**:
This pipeline implements a multioutput classifier model using random forest.  
Run the following commands in the *models* directory: 
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
**Web app**:
Run the following commands in the *app* directory:
```
python run.py
```
And go to http://localhost:5000/  to see how the project works at your local machine.


### Project Structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database with saved clean data

- models
|- train_classifier.py
|- classifier.pkl  # saved random forest model 

- requirements.txt
- README.md
```


### Acknowledgements

[Figure Eight](https://www.figure-eight.com/)  for providing the dataset 
