from flask import Flask, render_template, send_file, request, jsonify
import json
import pandas as pd
from model import getModel

app = Flask(__name__)
model = getModel()

@app.route('/')
def homepage():
    with open('e7_data/herocodes.json', 'r') as file:
        data = json.load(file)
    return render_template('homepage.html', hero_list =data)

@app.route('/image')
def get_vs():
    filename = 'VS.png'  # Name of the image file
    return send_file('images/' + filename)
 
@app.route('/calculateWin', methods=['POST'])
def calculateWin():
    data = request.json  # Read JSON data sent from client
    # Process the data here
    print(data)
    return "Endpoint called successfully"

if __name__ == '__main__':
    app.run()