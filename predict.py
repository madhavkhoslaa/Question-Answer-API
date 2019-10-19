from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
from operator import itemgetter
from keras.models import load_model
import pandas as pd
import pickle 
import json
from flask import Flask
import sys
import json


with open('Answers/qa_pair.json') as json_file:
    data = json.load(json_file)

def load_files():
	infile = open("Pickle/tokeniser.pkl",'rb')
	tokenizer = pickle.load(infile)
	infile.close()

	infile = open("Pickle/embedding_matrix.pkl",'rb')
	embedding_matrix = pickle.load(infile)
	infile.close()
	return tokenizer, embedding_matrix

tokenizer, embedding_matrix= load_files()
model = load_model('Weights/lstm_50_50_0.17_0.25.h5')

with open('Answers/qa_pair.json') as json_file:
    data = json.load(json_file)

test_sentence_pairs= [] #Stores statements as pairs in a tuple
def pred(test_sentence_pairs):
	test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])
	preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=0).ravel())
	results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
	results.sort(key=itemgetter(2), reverse=True)
	return results

question= sys.argv[1]
question= question.replace("-", " ")
for question_list in list(data.keys()):
	test_sentence_pairs.append((question, question_list))
	res= pred(test_sentence_pairs)
result_= dict()
for _ in res:
	orignal_question= _[0]
	question_inlisted= _[1]
	similarity_metric= _[2]

	result_[_[1]]= [{"Answer": data[_[1]]} , {"Similarity Metric": str(similarity_metric)}]

with open('most-recent-answer.json', 'w') as f:
    json.dump(result_, f)