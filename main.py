import re
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import seaborn as sns
from zipfile import ZipFile

def clean(text):
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[#]+|[^A-Za-z0-9]+"
    text_cleaning_hash = "#[A-Za-z0-9]+"
    text_cleaning_num = "(^|\W)\d+"
    text_cleaning_space = "xa0"
    text_cleaning_bacajuga = "baca juga"

    text = re.sub(text_cleaning_re, " ", str(text)).strip()
    text = re.sub(text_cleaning_hash, " ", str(text)).strip()
    text = re.sub(text_cleaning_num, " ", str(text)).strip()
    text = re.sub(text_cleaning_space, " ", str(text)).strip()
    text = re.sub(text_cleaning_bacajuga, " ", str(text)).strip()

    out = []
    for word in text.split():
        out.append(word)

    return out

#Input zip file name to be unzip
def un_zipFiles(unzip_path, file_name_concat):
    a = os.path.join(unzip_path,file_name_concat)
    zf = ZipFile(a, 'r')
    zf.extractall(unzip_path)
    zf.close()

file_name = input("file name to unzip : ")
zip_name = ".zip"
file_name_concat = file_name + zip_name
unzip_path = "your path"
un_zipFiles(unzip_path, file_name_concat)

basepath_banjir = f"your path//{file_name}//banjir//"
basepath_narkoba = f"your path//{file_name}//narkoba//"

#Load data
banjir = []
for entry in os.listdir(basepath_banjir):
    if os.path.isfile(os.path.join(basepath_banjir, entry)):
        banjir.append(entry)

berita_banjir =[]
for i in os.listdir(basepath_banjir):
    with open(basepath_banjir + i, 'r') as f:
        isi = f.readlines()
        isi_clean = clean(isi)
        berita_banjir.append(isi_clean)

narkoba = []
for entry in os.listdir(basepath_narkoba):
    if os.path.isfile(os.path.join(basepath_narkoba, entry)):
        narkoba.append(entry)

berita_narkoba =[]
for i in narkoba:
    with open(basepath_narkoba + i, 'r') as f:
        isi = f.readlines()
        isi_clean = clean(isi)
        berita_narkoba.append(isi_clean)

        
#Data Preprocessing
data_banjir = pd.DataFrame({'text':berita_banjir, 'category':'banjir'})
data_narkoba = pd.DataFrame({'text':berita_narkoba, 'category':'narkoba'})

df = pd.concat([data_narkoba, data_banjir])
df = df.reset_index().drop("index",axis = 1)
df['text'] = df['text'].apply(lambda x: x[0])


#Train-Test Split
x = df['text']
y = df['category']
x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)


#Initiate Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(x_train, y_train)
labels = model.predict(x_test)


#Model Evaluation (Accuracy Score)
names = np.unique(y)
print(classification_report(y_test, labels, target_names=names))


#Model Evaluation (Conf Matrix)
names = np.unique(y)
mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

#Model Evaluation (Test New Sentences)
def predict_category(s, model=model):
    pred = model.predict([s])
    return pred[0]

sentences = ["narkoba di jakarta"]
for i in sentences:
    print(i, ":", predict_category(i))
