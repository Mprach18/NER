from flask import Flask,render_template,url_for,request
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/process',methods=["POST"])
def process():   
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        ner=[]
        for sent in nltk.sent_tokenize(rawtext):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label'):
                    new=(chunk.label(), ' '.join(c[0] for c in chunk))
                    ner.append(new)
    
    return render_template("index.html",results=ner)


if __name__ == '__main__':
	app.run(debug=True)
