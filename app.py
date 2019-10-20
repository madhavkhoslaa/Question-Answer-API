from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
from operator import itemgetter
from keras.models import load_model
import pandas as pd
import pickle
import json
from flask import Flask, request, Response, render_template
import os
import time
import datetime
from datetime import datetime

app = Flask(__name__)
def load_files():
    infile = open("Pickle/tokeniser.pkl", 'rb')
    tokenizer = pickle.load(infile)
    infile.close()

    infile = open("Pickle/embedding_matrix.pkl", 'rb')
    embedding_matrix = pickle.load(infile)
    infile.close()
    return tokenizer, embedding_matrix


tokenizer, embedding_matrix = load_files()
model = load_model('Weights/lstm_50_50_0.17_0.25.h5')

with open('Answers/qa_pair.json') as json_file:
    data = json.load(json_file)
def read_file_send_content(file_loc):
    with open(file_loc, 'r') as content_file:
        content = content_file.read()
        return content
test_sentence_pairs = []  # Stores statements as pairs in a tuple
@app.route("/Answer/<string:question>", methods=["GET"])
def suggestedAnswers(question):
    init = time.time()
    query_details = dict()
    os.system("python3 predict.py {}".format(question))
    recent_file = open("most-recent-answer.json", 'r')
    contents = recent_file.read()
    os.system("rm -rf most-recent-answer.json")
    final = time.time()
    exec_time = final - init
    query_details["exec time"] = exec_time
    query_details["Endpoint"] = "Answer"
    query_details["question asked"] = question
    now = datetime.now() 
    query_details["date-time"] = now.strftime("%m/%d/%Y, %H:%M:%S")
    file = open("logs/log.txt", mode="a")
    file.write(str(query_details))
    file.write("\n")
    file.close
    return str(contents)


@app.route("/Server-Logs/<string:passcode>", methods=["GET"])
def send_logs_pls(passcode):
    with open('Answers/passcode.json') as json_file:
        data = json.load(json_file)
        corr_pass = data["Passcode"]
        if corr_pass == passcode:
            content = read_file_send_content("logs/log.txt")
            return {"Server Logs": content}
        else:
            return {"Cannot Access": "Permission Denied, Incorrect Password"}

@app.route("/qa-dict-update/<string:passcode>", methods=["POST"])
def QA_update(passcode):
        data_dict= request.get_json()
        print(data_dict)
        with open('Answers/passcode.json') as json_file:
            data = json.load(json_file)
            corr_pass = data["Passcode"]
            if corr_pass == passcode:
                with open('Answers/qa_pair.json', 'w') as f:
                    json.dump(data_dict, f)
                return {"Updated": data_dict}
            else:
                return {"Cannot Access": "Permission Denied, Incorrect Password"}
@app.route("/help", methods=["GET"])
def help_():
    return render_template("help.html")

@app.route("/", methods=["GET"])
def slash():
    return {"To get help go to /help endpoint": ""}
    


if __name__ == "__main__":
    app.run(threaded=True, debug=True, host= "0.0.0.0")
