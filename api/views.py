# api/views.py
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
import json
import re
import gensim
import emoji
from pyvi import ViTokenizer
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd
import numpy as np
from api.apps import ApiConfig  # Import ApiConfig

def Preprocess(string):
#Remove extended characters Ex:đẹppppppp
    string = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), string, flags=re.IGNORECASE)
#Remove special characters
    string = gensim.utils.simple_preprocess(string)
    string = ' '.join(string)
    string = re.sub(r"[-()\\\"#/@;:<>{}`+=~|.!?,%/]", "",string)
    string = re.sub('\n', ' ',string)
    string = re.sub('--', '',string)
    string = re.sub('  ', ' ',string)
    string = re.sub('   ', ' ',string)
    string = re.sub('    ', ' ',string)
#lowercase
    string = string.lower()
#Remove link
    string = re.sub('<.*?>', '', string).strip()
    string = re.sub('(\s)+', r'\1', string)
#Remove number
    string = re.sub(r"\d+", "", string)
#Remove useless English phrases
    string = re.sub("added.*photo", "", string)
    string = re.sub("added.*photos", "", string)
    string = re.sub("is.*post", "", string)
    string = re.sub("Photos.*post", "", string)
    string = re.sub("from.*post", "", string)
    string = re.sub("shared.*group", "", string)
    string = re.sub("shared.*post", "", string)
    string = re.sub("shared.*video", "", string)
    string = re.sub("is.*motivated", "", string)
    string = re.sub("is.*with", "", string)

#Remove redundant characters
    string = string.replace(u'"', u' ')
    string = string.replace(u'️', u'')
    string = string.replace('🏻','')
    string = string.replace('_',' ')
    string = string.replace('—',' ')
    string = " ".join(string.split())
    string = string.replace('url','')
    return string

#Remove emoji
def give_emoji_free_text(text):
    cleaned_text = emoji.demojize(text)
    return cleaned_text

def word_segment(string):
    string=ViTokenizer.tokenize(string)
    return string

max_len = 220
model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.hdf5')
loaded_model = load_model(model_path)
file_path = os.path.join(os.path.dirname(__file__), 'data', '02_train_text.csv')

# dataTrainText = pd.read_csv(file_path, header = 0, names = ['id', 'free_text'])
# dataTrainText['free_text'] = (
#     dataTrainText['free_text']
#     .str.lower()  # Convert to lowercase
#     .apply(Preprocess)  # Apply Preprocess function
#     .apply(give_emoji_free_text)  # Apply give_emoji_free_text function
#     .apply(word_segment)  # Apply word_segment function
# )
# train_descs = dataTrainText['free_text']
# print(train_descs[0])
# print(train_descs[1])
# output_file_path = 'train_descs.txt'

# Save train_descs to a text file
# train_descs.to_csv(output_file_path, index=False, header=False)
# embedding_path = os.path.join(os.path.dirname(__file__), 'fasttest', 'cc.vi.300.vec')
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data', 'train_descs.txt')
# with open(file_path, 'r', encoding='utf-8') as file:
#     train_descs = file.read()

with open(file_path, 'r', encoding='utf-8') as file:
    train_descs = [line.strip() for line in file.readlines()]
print(train_descs[0])
print(train_descs[1])
print(train_descs[2])
embed_size = 300
max_features = 130000
tk = Tokenizer(num_words=max_features, lower=True)
tk.fit_on_texts(train_descs)
# def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
# embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf-8"))       
# word_index = tk.word_index
# nb_words = min(max_features, len(word_index) + 1)
# embedding_matrix = np.zeros((nb_words, embed_size))
# for word, i in word_index.items():
#     if i >= max_features: continue
#     embedding_vector = embedding_index.get(word)
#     print(embedding_vector)
#     if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def detect_hate_speech(text): 
    text_to_predict = [text]
    print(text_to_predict)
    for i in range(len(text_to_predict)):
        text_to_predict[i] = text_to_predict[i].lower()
        text_to_predict[i] = Preprocess(text_to_predict[i])
        text_to_predict[i] = give_emoji_free_text(text_to_predict[i])
        text_to_predict[i] = word_segment(text_to_predict[i])
    sequences_to_predict =tk.texts_to_sequences(text_to_predict)
    padded_sequences_to_predict = pad_sequences(sequences_to_predict, maxlen=max_len)
    probabilities_to_predict = loaded_model.predict(padded_sequences_to_predict)
    predicted_class_index_to_predict = probabilities_to_predict.argmax(axis=-1)
    predicted_class_label_to_predict = predicted_class_index_to_predict[0]
    if (predicted_class_label_to_predict != 0):
        return True
    return False

@csrf_exempt
def hate_speech_detection(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            text = data.get('text', '')
            if text:
                is_hate_speech = detect_hate_speech(text)
                response_data = {'is_hate_speech': is_hate_speech}
                return JsonResponse(response_data, status=200)
            else:
                return JsonResponse({'error': 'Text parameter is required'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
