from dog_breed_detector import detect_dog_breed

import json
import sqlite3
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify, url_for
from plotly.graph_objs import Bar, Pie, Heatmap
from sqlalchemy import create_engine
import os
import re


UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


try:
    print('prerforming final setup...')
    human_or_dog, dog_breed = detect_dog_breed('./test_images/human2.jpg')
except:
    human_or_dog, dog_breed = '',''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_data_from_db_and_create_db_if_not_exist(path):
    try:
        print('trying to find db')
        engine = create_engine('sqlite:///'+path)
        df = pd.read_sql_table('classification_history', engine)
        print('db found!')
    except:
        print('db not found... creating new!')
        conn = sqlite3.connect(path)
        # get a cursor
        cur = conn.cursor()
        # drop the test table in case it already exists
        cur.execute("DROP TABLE IF EXISTS classification_history")
        # create the test table including project_id as a primary key
        cur.execute("CREATE TABLE classification_history ( Output_Classification TEXT);")
        # insert a value into the classification_history table
        cur.execute('INSERT INTO classification_history (Output_Classification) VALUES ( "{}");'.format(''))
        #cur.execute('INSERT INTO classification_history (input_text, Output_Classification) VALUES ( "{}", "{}");'.format(query,  str(calassification_results)))
        conn.commit()
        # commit any changes and close the data base
        conn.close()
        df = pd.read_sql_table('classification_history', engine)
    return df


def save_classification_results(classification_label,db_historical_classifications_path):
    try:
        conn = sqlite3.connect(db_historical_classifications_path)
        cur = conn.cursor()
        # drop the test table in case it already exists
        cur.execute('INSERT INTO classification_history (Output_Classification) VALUES ( "{}");'.format(str(classification_label)))
        conn.commit()      
        conn.close()
        return 1
    except:
        return 0



# path of historical predictions db
db_historical_classifications_path = './historic_predictions/historical_predictions.db'


# index webpage displays cool visuals and receives user input text for model
@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
 

    
    # : load data from historical_predictions.db. if not present, create one.
    # Data will be kept in table named classification_history
    # Table wil have attributes Output_Classification:string (string containing classification label )
    df_historical_classifications = load_data_from_db_and_create_db_if_not_exist(db_historical_classifications_path)
    #print(df_historical_classifications.Output_Classification.value_counts())
    temp = df_historical_classifications.Output_Classification.value_counts()
    historical_classifications_counts = temp
    historical_classifications_names = list(temp.index)
    
  

    

    
    # create visuals
    # : 
    graphs = [
        # : Add graph for historical prediction counts
         {
            'data': [
                Bar(
                    x=historical_classifications_names,
                    y=historical_classifications_counts
                )
            ],

            'layout': {
                'title': 'Counts of Historical Classification Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    #print(ids)
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    print('3##########')
    output_string = ''
    path = ''
    if request.method == 'POST':
            if 'file1' not in request.files:
                print( 'there is no file1 in form!')
            file1 = request.files['file1']
            if file1.filename == '': # if no file selected
                print('No selected file')
                return render_template('master.html',img_path =  path,output_string = output_string, ids=ids, graphJSON=graphJSON)
            else:
                path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
                file1.save(path)
                print('file uploaded to :', path)

                print('Applying Classification model...')
                human_or_dog, dog_breed = detect_dog_breed(path)

                if human_or_dog == 'Unknown' and dog_breed == 'Unknown':
                    output_string = 'No Human or Dog found in image...'
                    print(output_string)
                else:
                    output_string = human_or_dog + ' found in image belonging to ' + dog_breed +  ' breed!'
                    print( output_string)
                if save_classification_results(dog_breed, db_historical_classifications_path):
                    print('classification results saved to db')
                else:
                    print('error!...couldnt save classification results saved to db')
            


#'http://127.0.0.1:3001/' +

    return render_template('master.html',img_path =  path,output_string = output_string, ids=ids, graphJSON=graphJSON)




def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()



