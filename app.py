from flask import Flask, render_template
import json

app = Flask(__name__)

@app.route('/')
def homepage():
    with open('e7_data/heronames.json', 'r') as file:
        data = json.load(file)
    return render_template('homepage.html', hero_list =data)
 
if __name__ == '__main__':
    app.run()