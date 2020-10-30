from flask import Flask, render_template, request

from fashion_detector import Fashion

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
  fashion = Fashion()
  cloth = fashion.predict_image(request.form['url'])
  return render_template('reponse.html', cloth=cloth)
      
if __name__ == '__main__':
  app.run()
  pass
