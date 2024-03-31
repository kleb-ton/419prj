from flask import Flask, render_template, send_file
import json
import pandas as pd

app = Flask(__name__)

@app.route('/')
def homepage():
    with open('e7_data/herocodes.json', 'r') as file:
        data = json.load(file)
    return render_template('homepage.html', hero_list =data)

@app.route('/image')
def get_vs():
    filename = 'VS.png'  # Name of the image file
    return send_file('images/' + filename)
 
if __name__ == '__main__':
    app.run()