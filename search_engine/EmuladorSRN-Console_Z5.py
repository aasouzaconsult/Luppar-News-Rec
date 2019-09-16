# - Só para bases próprias (Z5News* e Z5NewsBrasil)
# - Tem como quando escolher a coleção, só mostrar as categorias daquela coleção?

#sportsNews (N - maiusculo)
#politicsNews
#technologyNews
#PersonalFinance
#brazil-news

###############################################
from engine.util import analyzer_nothing, E2V_IDF

print ("Importando Bibliotecas e Classes...") #
###############################################

from time import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gensim
import pickle
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn import tree    # https://scikit-learn.org/stable/modules/tree.html#
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn import metrics
from sklearn.utils.fixes import signature
from sklearn.preprocessing import label_binarize

####################################
print ("Classes Embeddings Médio") #
####################################
# Calcula a média dos vetores de cada uma das palavras do documento - para cada um dos documentos
#
# class E2V_AVG(object):
#     def __init__(self, word2vec):
#         self.w2v = word2vec
#         self.dimensao = 300
#
#     def fit(self, X, y):
#         return self
#
#     def transform(self, X):
#         return np.array([
#             np.mean([self.w2v[word] for word in words if word in self.w2v] or [np.zeros(self.dimensao)], axis=0)
#             for words in X
#         ])
#
# ###################################################
# print ("Classe da Abordagem Proposta - E2V-IDF*") #
# ###################################################
#
# class E2V_IDF(object):
#     def __init__(self, word2vec):
#         self.w2v = word2vec
#         self.wIDF = None # IDF da palavra na colecao
#         self.dimensao = 300
#
#     def fit(self, X, y):
#         tfidf = TfidfVectorizer(analyzer=lambda x: x)
#         tfidf.fit(X)
#         maximo_idf = max(tfidf.idf_) # Uma palavra que nunca foi vista (rara) então o IDF padrão é o máximo de idfs conhecidos (exemplo: 9.2525763918954524)
#         self.wIDF = defaultdict(
#             lambda: maximo_idf,
#             [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
#         return self
#
#     # Gera um vetor de 300 dimensões, para cada documento, com a média dos vetores (embeddings) dos termos * IDF, contidos no documento.
#     def transform(self, X):
#         return np.array([
#                 np.mean([self.w2v[word] * self.wIDF[word] for word in words if word in self.w2v] or [np.zeros(self.dimensao)], axis=0)
#                 for words in X
#             ])

###################################################
print ("Importando as coleções de documentos...") #
###################################################

# Stemmed
X = pickle.load(open('data/reuters_X_pp_new.ipy', 'rb'))
y = pickle.load(open('data/reuters_y_pp_new.ipy', 'rb'))
z = pickle.load(open('data/reuters_z_pp_new.ipy', 'rb'))

X, y = np.array(X), np.array(y)

print ("Total de notícias - Reuters:  %s" % len(y)) # 42532


######################
print ("EMBEDDINGS") #
######################

print("-> Word2Vec - GENSIM") # (GENSIM) - https://radimrehurek.com/gensim/models/word2vec.html
#model = Word2Vec(X, size=300, window=5, sg=1, workers=4) # sg=1 Skip Gram
#w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}       # 8978

#pickle.dump(w2v, open('data/w2v_z5.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
w2v  = pickle.load(open('data/emb/w2v_z5.ipy', 'rb')) #(ship-gram - 300 - word2vec)

print("-> FastText - GENSIM") # (GENSIM) - https://radimrehurek.com/gensim/models/fasttext.html
#model_ft = FastText(X, size=300, window=5, sg=1, workers=4) # sg=1 Skip Gram
#ft  = {w: vec for w, vec in zip(model_ft.wv.index2word, model_ft.wv.syn0)} # 8978

#pickle.dump(ft, open('data/ft_z5.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
ft  = pickle.load(open('data/emb/ft_z5.ipy', 'rb')) #(ship-gram - 300 - fastText)

#####################
print ("PIPELINES") #
#####################
svm_rbf_bow   = Pipeline([("count_vectorizer", CountVectorizer(analyzer=analyzer_nothing)), ("svm rbf bow"  , OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0)))])
rf_bow        = Pipeline([("count_vectorizer", CountVectorizer(analyzer=analyzer_nothing)), ("rf bow"  , OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])

#----------------------------------#
print ("-> Embeddings - Word2Vec") #
#----------------------------------#
svm_rbf_w2v_idf = Pipeline([("w2v-idf", E2V_IDF(w2v)), ("svm rbf w2v-idf", OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])
rf_w2v_idf      = Pipeline([("w2v-idf", E2V_IDF(w2v)), ("rf w2v-idf", OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])

#----------------------------------#
print ("-> Embeddings - FastText") #
#----------------------------------#
svm_rbf_ft_idf  = Pipeline([("ft-idf", E2V_IDF(ft)), ("svm rbf ft-idf", OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])                           
rf_ft_idf       = Pipeline([("ft-idf", E2V_IDF(ft)), ("rf ft-idf", OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])


############################         TESTE - 1 DIA         ############################
#######################################################################################
############################ TESTE COM OUTRAS FONTES (SRN) ############################
#######################################################################################

# Test - Diversas fontes (Reuters e Inshorts) - 1 dia
#####################################################
# Stemmed
Xt = pickle.load(open('data/reuters_X_Test_pp_new_1d.ipy', 'rb'))
yt = pickle.load(open('data/reuters_y_Test_pp_new_1d.ipy', 'rb'))
zt = pickle.load(open('data/reuters_z_Test_pp_new_1d.ipy', 'rb'))

Xt, yt, zt = np.array(Xt), np.array(yt), np.array(zt)

print ("Z5News (Teste Online) - Total de notícias curtas: %s" % len(yt))

# Training and Testing Sets
#################################################################
#Z5 - BR
name_labels = ['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews']
Y           = label_binarize(y,  classes=['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews'])
Yt          = label_binarize(yt, classes=['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews'])
n_classes = Y.shape[1]

X_train = X
Y_train = Y
Z_train = z

X_test  = Xt
Y_test  = Yt
Z_test  = zt

################################
print ("Treinamento e Predição")
################################

###############################
# USAR NO BAIXAR E RECOMENDAR #
###############################
print ("Treinamento Modelo para a Z5News...")
t0 = time()
print ("svm_rbf_ft_idf")
svm_rbf_ft_idf.fit(X_train, Y_train)  # 666.894s
print ("svm_rbf_w2v_idf")
svm_rbf_w2v_idf.fit(X_train, Y_train)  # 667.059s
print ("svm_rbf_bow")
svm_rbf_bow.fit(X_train, Y_train) # 2273.284s

print ("rf_ft_idf")
rf_ft_idf.fit(X_train, Y_train) # 58.044s
print ("rf_w2v_idf")
rf_w2v_idf.fit(X_train, Y_train) # 58.382s
print ("rf_bow")
rf_bow.fit(X_train, Y_train) # 11.557s

# Salvar Models
pickle.dump(svm_rbf_ft_idf , open('model/z5_model_svm_ft.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(svm_rbf_w2v_idf, open('model/z5_model_svm_w2v.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(svm_rbf_bow, open('model/z5_model_svm_bow.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)

pickle.dump(rf_ft_idf, open('model/z5_model_rf_ft.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(rf_w2v_idf, open('model/z5_model_rf_w2v.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(rf_bow, open('model/z5_model_rf_bow.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)


#print ("Predição...")
#z5_pred_svm_ft  = svm_rbf_ft_idf.predict(X_test)
#z5_pred_svm_w2v = svm_rbf_w2v_idf.predict(X_test)
#z5_pred_svm_bow = svm_rbf_bow.predict(X_test)
#
#z5_pred_rf_ft  = rf_ft_idf.predict(X_test)
#z5_pred_rf_w2v = rf_w2v_idf.predict(X_test)
#z5_pred_rf_bow = rf_bow.predict(X_test)
#
## Salvar predições
#pickle.dump(z5_pred_svm_ft, open('data/pred/z5_pred_svm_ft.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
#pickle.dump(z5_pred_svm_w2v, open('data/pred/z5_pred_svm_w2v.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
#pickle.dump(z5_pred_svm_bow, open('data/pred/z5_pred_svm_bow.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
#
#pickle.dump(z5_pred_rf_ft, open('data/pred/z5_pred_rf_ft.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
#pickle.dump(z5_pred_rf_w2v, open('data/pred/z5_pred_rf_w2v.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
#pickle.dump(z5_pred_rf_bow, open('data/pred/z5_pred_rf_bow.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)


# Recuperando predicoes
# z5_pred_svm_ft   = pickle.load(open('data/pred/z5_pred_svm_ft.ipy', 'rb'))
# z5_pred_svm_w2v  = pickle.load(open('data/pred/z5_pred_svm_w2v.ipy', 'rb'))
# z5_pred_svm_bow  = pickle.load(open('data/pred/z5_pred_svm_bow.ipy', 'rb'))
#
# z5_pred_rf_ft   = pickle.load(open('data/pred/z5_pred_rf_ft.ipy', 'rb'))
# z5_pred_rf_w2v  = pickle.load(open('data/pred/z5_pred_rf_w2v.ipy', 'rb'))
# z5_pred_rf_bow  = pickle.load(open('data/pred/z5_pred_rf_bow.ipy', 'rb'))
#
# # Exemplo
# predictions = z5_pred_svm_ft
#
# print ("Calculando métricas...")
# print ("Precision: %s" % precision_score(Y_test, predictions, average="micro"))
# print ("Recall...: %s" % recall_score(Y_test, predictions, average="micro"))
# print ("F1-Score.: %s" % f1_score(Y_test, predictions, average="micro"))
# print ("Accuracy.: %s" % accuracy_score(Y_test, predictions))
#
# #print (classification_report(Y_test, predictions))
#
# # RMSE
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# rmse = sqrt(mean_squared_error(Y_test, predictions))
# print("RMSE: ",rmse)
#
# print ("Classificador SVM - Documentos Retornados...")
# docsRetornados = []
# lblsRetornados = []
# result = 0
#
# # Salvando os documentos e labels previstos...
# for j in range(len(predictions)):
#    for i in range(n_classes):
#       if predictions[j][i] == 1:
#          idx = i
#          docsRetornados.append(Z_test[j])
#          lblsRetornados.append(name_labels[idx])
#
# # Retornando - TELA DO SRN
# print ("Classificador SVM - Consultando 1 ou 2 Tópicos...")
# for i in range(len(lblsRetornados)):
#   #if lblsRetornados[i] == 'sportsNews' or lblsRetornados[i] == 'politicsNews':
#    if lblsRetornados[i] == 'sportsNews':
#       result = result + 1
#       print (lblsRetornados[i])
#       print (docsRetornados[i])
#
# print ("-------------------------------------------------")
# print ("Documentos Recuperados: %s" % result)
# print (classification_report(Y_test, predictions, target_names = name_labels))
# print ("-------------------------------------------------")

#########################################
##### TESTE - Visualizar o Embedding ####
#########################################
## TSE
#from sklearn.manifold import TSNE
#
#model_ft.similar_by_word('computer')
#
#def display_closestwords_tsnescatterplot(model, word):
#    
#    arr = np.empty((0,300), dtype='f')
#    word_labels = [word]
#
#    # get close words
#    close_words = model.similar_by_word(word, 40)
#    
#    # add the vector for each of the closest words to the array
#    arr = np.append(arr, np.array([model[word]]), axis=0)
#    for wrd_score in close_words:
#        wrd_vector = model[wrd_score[0]]
#        word_labels.append(wrd_score[0])
#        arr = np.append(arr, np.array([wrd_vector]), axis=0)
#        
#    # find tsne coords for 2 ou 3 dimensions
#    tsne = TSNE(n_components=3, random_state=0)
#    np.set_printoptions(suppress=True)
#    Y = tsne.fit_transform(arr)
#
#    x_coords = Y[:, 0]
#    y_coords = Y[:, 1]
#    # display scatter plot
#    plt.scatter(x_coords, y_coords)
#
#    for label, x, y in zip(word_labels, x_coords, y_coords):
#        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
#    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
#    plt.show()
#
#display_closestwords_tsnescatterplot(model_ft, 'computer')
#display_closestwords_tsnescatterplot(model_ft, 'apple')
#display_closestwords_tsnescatterplot(model_ft, 'nba')
#
#######################################################
## PCA
#from sklearn.decomposition import PCA
#from matplotlib import pyplot
#
#X = model_ft[model_ft.wv.vocab]
#
#pca = PCA(n_components=2)
#result = pca.fit_transform(X)
#
## create a scatter plot of the projection
#pyplot.scatter(result[:, 0], result[:, 1])
#words = list(model_ft.wv.vocab)
#for i, word in enumerate(words):
#    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#pyplot.show()
#
#######################################################
## Testes
#result = model_ft.most_similar(positive=['woman', 'nba'], negative=['man'])
#print(result)