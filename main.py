import pickle
from flask import Flask , request, jsonify
from flask.wrappers import Response
from model_files.bert_model import answering


app = Flask("qa_model")

@app.route('/', methods=['POST'])
def ping():
    y= request.get_json()
    question=y['question']

   
    
  

    ans=answering(question)

    

    return jsonify(ans)





if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=9696)
