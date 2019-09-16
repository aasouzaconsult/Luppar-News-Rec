###############################################
print ("Importando Bibliotecas e Classes...") #
###############################################

import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

###################################################
print ("Importando as coleções de documentos...") #
###################################################

# Stemmed
X = pickle.load(open('data/g1_X_Z5BR_pp.ipy', 'rb'))
y = pickle.load(open('data/g1_y_Z5BR_pp.ipy', 'rb'))
z = pickle.load(open('data/g1_z_Z5BR_pp.ipy', 'rb'))

X, y, z = np.array(X), np.array(y), np.array(z)

print ("Z5NewsBrasil - Total de notícias curtas - Train: %s" % len(y))

# Test - Diversas fontes (agregador - UOL)
###################################################
# Stemmed
Xt = pickle.load(open('data/g1_X_Z5BR_pp_Test_1d.ipy', 'rb'))
yt = pickle.load(open('data/g1_y_Z5BR_pp_Test_1d.ipy', 'rb'))
zt = pickle.load(open('data/g1_z_Z5BR_pp_Test_1d.ipy', 'rb'))

Xt, yt, zt = np.array(Xt), np.array(yt), np.array(zt)

print ("Z5NewsBrasil (Teste Online) - Total de notícias curtas: %s" % len(yt))

# Training and Testing Sets
#################################################################
#Z5 - BR
name_labels = ['esporteNews', 'politicaNews', 'tecnologiaNews', 'financaPessoal', 'educacaonews']
Y           = label_binarize(y,  classes=['esporteNews', 'politicaNews', 'tecnologiaNews', 'financaPessoal', 'educacaonews'])
Yt          = label_binarize(yt, classes=['esporteNews', 'politicaNews', 'tecnologiaNews', 'financaPessoal', 'educacaonews'])
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

# Recuperando predicoes
print ("Importando Modelo - combinação: SVM(RBF) combinado com FastText utilizando a abordagem E2V-IDF...")
z5br_pred_svm_ft   = pickle.load(open('data/pred/z5br_pred_svm_ft.ipy', 'rb'))

# Exemplo
predictions = z5br_pred_svm_ft

print ("Calculando métricas...")
print ("Precision: %s" %precision_score(Y_test, predictions, average="micro"))
print ("Recall...: %s" %recall_score(Y_test, predictions, average="micro"))
print ("F1-Score.: %s" %f1_score(Y_test, predictions, average="micro"))
print ("Accuracy.: %s" %accuracy_score(Y_test, predictions))

# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(Y_test, predictions))
print("RMSE: ",rmse)

print ("Classificador SVM - Documentos Retornados...")
docsRetornados = []
lblsRetornados = []
result = 0

# Salvando os documentos e labels previstos...
for j in range(len(predictions)):
   for i in range(n_classes):
      if predictions[j][i] == 1:
         idx = i
         docsRetornados.append(Z_test[j])
         lblsRetornados.append(name_labels[idx])

# Retornando - TELA DO SRN
print ("Classificador SVM - Consultando 1 ou 2 Tópicos...")
for i in range(len(lblsRetornados)):
   if lblsRetornados[i] == 'educacaonews': # politicaNews, tecnologiaNews, financaPessoal, educacaonews
      result = result + 1
      print (lblsRetornados[i])
      print (docsRetornados[i])

print ("-------------------------------------------------")
print ("Documentos Recuperados: %s" % result)
print (classification_report(Y_test, predictions, target_names = name_labels))
print ("-------------------------------------------------")
