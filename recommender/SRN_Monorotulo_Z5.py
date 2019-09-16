###############################################
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

class E2V_AVG(object):
    def __init__(self, word2vec):
        self.w2v = word2vec
        self.dimensao = 300
    
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.w2v[word] for word in words if word in self.w2v] or [np.zeros(self.dimensao)], axis=0)
            for words in X
        ])

###################################################
print ("Classe da Abordagem Proposta - W2V-IDF*") #
###################################################

class E2V_IDF(object):
    def __init__(self, word2vec):
        self.w2v = word2vec
        self.wIDF = None # IDF da palavra na colecao
        self.dimensao = 300
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        maximo_idf = max(tfidf.idf_) # Uma palavra que nunca foi vista (rara) então o IDF padrão é o máximo de idfs conhecidos (exemplo: 9.2525763918954524)
        self.wIDF = defaultdict(
            lambda: maximo_idf, 
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
        return self
    
    # Gera um vetor de 300 dimensões, para cada documento, com a média dos vetores (embeddings) dos termos * IDF, contidos no documento.
    def transform(self, X):
        return np.array([
                np.mean([self.w2v[word] * self.wIDF[word] for word in words if word in self.w2v] or [np.zeros(self.dimensao)], axis=0)
                for words in X
            ])

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
model = Word2Vec(X, size=300, window=5, sg=1, workers=4) # sg=1 Skip Gram
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}       # 8713

print("-> FastText - GENSIM") # (GENSIM) - https://radimrehurek.com/gensim/models/fasttext.html
model_ft = FastText(X, size=300, window=5, sg=1, workers=4) # sg=1 Skip Gram
ft  = {w: vec for w, vec in zip(model_ft.wv.index2word, model_ft.wv.syn0)} # 8713

# ---------------------------------------------
print("-> Word2Vec - WIKI 2019 (pré-treinado)") 
w2v_wiki = pickle.load(open('data/emb_wiki_sg_300_word2vec.ipy', 'rb')) # 1494957 (ship-gram - 300)

print("-> FastText - WIKI 2019 (pré-treinado)") 
ft_wiki  = pickle.load(open('data/emb_wiki_sg_300_fasttext.ipy', 'rb')) # 1494957 (ship-gram - 300 - fastText)


#####################
print ("PIPELINES") #
#####################

#-----------------#
print ("-> Pure") #
#-----------------#
# SVM + RBF (Support Vector Machine + Radial Basis Function)
# y = gamma (grau da Curva, < + reto e > - reto ) | C = largura da margem do Hiperplano
svm_rbf_bow   = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("svm rbf bow"  , OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0)))])
svm_rbf_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("svm rbf tfidf", OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0)))])
# com n_jobs=-1 - ValueError: WRITEBACKIFCOPY base is read-only

# KNN - K-Nearest Neighbors
knn_bow   = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("knn bow"  , OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
knn_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("knn tfidf", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
# p - distancia (1 = manhattan, 2 = euclidean* e 3 = minkowski...)

# Decision Tree
dt_bow   = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("dt bow"  , OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])
dt_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("dt tfidf", OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])

# Random Forest (teste)
rf_bow   = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("rf bow"  , OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])
rf_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("rf tfidf", OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])

#----------------------------------#
print ("-> Embeddings - Word2Vec") #
#----------------------------------#
svm_rbf_w2v     = Pipeline([("w2v", E2V_AVG(w2v))    , ("svm rbf w2v",     OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])
svm_rbf_w2v_idf = Pipeline([("w2v-idf", E2V_IDF(w2v)), ("svm rbf w2v-idf", OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])

knn_w2v         = Pipeline([("w2v", E2V_AVG(w2v))    , ("knn w2v",     OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
knn_w2v_idf     = Pipeline([("w2v-idf", E2V_IDF(w2v)), ("knn w2v-idf", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])

dt_w2v       = Pipeline([("w2v", E2V_AVG(w2v))    , ("dt w2v",     OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])
dt_w2v_idf   = Pipeline([("w2v-idf", E2V_IDF(w2v)), ("dt w2v-idf", OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])

rf_w2v       = Pipeline([("w2v", E2V_AVG(w2v))    , ("rf w2v",     OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])
rf_w2v_idf   = Pipeline([("w2v-idf", E2V_IDF(w2v)), ("rf w2v-idf", OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])

#--------------------------------------------------------#
print ("-> Embeddings - Word2Vec (Wiki - pré-treinado)") #
#--------------------------------------------------------#
svm_rbf_w2v_wiki     = Pipeline([("w2v_wiki", E2V_AVG(w2v_wiki))    , ("svm rbf w2v_wiki",     OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])
svm_rbf_w2v_idf_wiki = Pipeline([("w2v-idf_wiki", E2V_IDF(w2v_wiki)), ("svm rbf w2v-idf_wiki", OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])

knn_w2v_wiki         = Pipeline([("w2v_wiki", E2V_AVG(w2v_wiki))    , ("knn w2v_wiki",     OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
knn_w2v_idf_wiki     = Pipeline([("w2v-idf_wiki", E2V_IDF(w2v_wiki)), ("knn w2v-idf_wiki", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])

dt_w2v_wiki       = Pipeline([("w2v_wiki", E2V_AVG(w2v_wiki))    , ("dt w2v_wiki",     OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])
dt_w2v_idf_wiki   = Pipeline([("w2v-idf_wiki", E2V_IDF(w2v_wiki)), ("dt w2v-idf_wiki", OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])

rf_w2v_wiki       = Pipeline([("w2v_wiki", E2V_AVG(w2v_wiki))    , ("rf w2v_wiki",     OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])
rf_w2v_idf_wiki   = Pipeline([("w2v-idf_wiki", E2V_IDF(w2v_wiki)), ("rf w2v-idf_wiki", OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])

#----------------------------------#
print ("-> Embeddings - FastText") #
#----------------------------------#
svm_rbf_ft     = Pipeline([("ft", E2V_AVG(ft))    , ("svm rbf ft",     OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])
svm_rbf_ft_idf = Pipeline([("ft-idf", E2V_IDF(ft)), ("svm rbf ft-idf", OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])
                             
knn_ft         = Pipeline([("ft", E2V_AVG(ft))    , ("knn ft",     OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
knn_ft_idf     = Pipeline([("ft-idf", E2V_IDF(ft)), ("knn ft-idf", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
                             
dt_ft       = Pipeline([("ft", E2V_AVG(ft))    , ("dt ft",     OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])
dt_ft_idf   = Pipeline([("ft-idf", E2V_IDF(ft)), ("dt ft-idf", OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])
                             
rf_ft       = Pipeline([("ft", E2V_AVG(ft))    , ("rf ft",     OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])
rf_ft_idf   = Pipeline([("ft-idf", E2V_IDF(ft)), ("rf ft-idf", OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])

#--------------------------------------------------------#
print ("-> Embeddings - FastText (Wiki - pré-treinado)") #
#--------------------------------------------------------#
svm_rbf_ft_wiki     = Pipeline([("ft_wiki", E2V_AVG(ft_wiki))    , ("svm rbf ft_wiki",     OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])
svm_rbf_ft_idf_wiki = Pipeline([("ft-idf_wiki", E2V_IDF(ft_wiki)), ("svm rbf ft-idf_wiki", OneVsRestClassifier(SVC(kernel="rbf", gamma=0.01, C=1.0), n_jobs=-1))])
                             
knn_ft_wiki         = Pipeline([("ft_wiki", E2V_AVG(ft_wiki))    , ("knn ft_wiki",     OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
knn_ft_idf_wiki     = Pipeline([("ft-idf_wiki", E2V_IDF(ft_wiki)), ("knn ft-idf_wiki", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, p=2)))])
                             
dt_ft_wiki       = Pipeline([("ft_wiki", E2V_AVG(ft_wiki))    , ("dt ft_wiki",     OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])
dt_ft_idf_wiki   = Pipeline([("ft-idf_wiki", E2V_IDF(ft_wiki)), ("dt ft-idf_wiki", OneVsRestClassifier(tree.DecisionTreeClassifier(min_samples_split=40), n_jobs=-1))])
                             
rf_ft_wiki       = Pipeline([("ft_wiki", E2V_AVG(ft_wiki))    , ("rf ft_wiki",     OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])
rf_ft_idf_wiki   = Pipeline([("ft-idf_wiki", E2V_IDF(ft_wiki)), ("rf ft-idf_wiki", OneVsRestClassifier(RandomForestClassifier(min_samples_split=40, n_estimators=10, n_jobs=-1), n_jobs=-1))])

all_models_svm = [
    ("SVM(RBF)+BoW", svm_rbf_bow),
    ("SVM(RBF)+TFIDF", svm_rbf_tfidf),
    ("SVM(RBF)+W2V", svm_rbf_w2v),
    ("SVM(RBF)+W2V-IDF", svm_rbf_w2v_idf),
    ("SVM(RBF)+FT", svm_rbf_ft),
    ("SVM(RBF)+FT-IDF", svm_rbf_ft_idf),
    # Wiki
    ("SVM(RBF)+W2V_WIKI", svm_rbf_w2v_wiki),
    ("SVM(RBF)+W2V-IDF_WIKI", svm_rbf_w2v_idf_wiki),
    ("SVM(RBF)+FT_WIKI", svm_rbf_ft_wiki),
    ("SVM(RBF)+FT-IDF_WIKI", svm_rbf_ft_idf_wiki),
]

all_models_knn = [
    ("KNN+BoW", knn_bow),
    ("KNN+TFIDF", knn_tfidf),
    ("KNN+W2V", knn_w2v),
    ("KNN+W2V-IDF", knn_w2v_idf),
    ("KNN+FT", knn_ft),
    ("KNN+FT-IDF", knn_ft_idf),
    # Wiki
    ("KNN+W2V_WIKI", knn_w2v_wiki),
    ("KNN+W2V-IDF_WIKI", knn_w2v_idf_wiki),
    ("KNN+FT_WIKI", knn_ft_wiki),
    ("KNN+FT-IDF_WIKI", knn_ft_idf_wiki),
]

all_models_dt = [
    ("DT+BoW", dt_bow),
    ("DT+TFIDF", dt_tfidf),
    ("DT+W2V", dt_w2v),
    ("DT+W2V-IDF", dt_w2v_idf),
    ("DT+FT", dt_ft),
    ("DT+FT-IDF", dt_ft_idf),
    # Wiki
    ("DT+W2V_WIKI", dt_w2v_wiki),
    ("DT+W2V-IDF_WIKI", dt_w2v_idf_wiki),
    ("DT+FT_WIKI", dt_ft_wiki),
    ("DT+FT-IDF_WIKI", dt_ft_idf_wiki),
]

all_models_rf = [
    ("RF+BoW", rf_bow),
    ("RF+TFIDF", rf_tfidf),
    ("RF+W2V", rf_w2v),
    ("RF+W2V-IDF", rf_w2v_idf),
    ("RF+TF", rf_ft),
    ("RF+TF-IDF", rf_ft_idf),
    # Wiki
    ("RF+W2V_WIKI", rf_w2v_wiki),
    ("RF+W2V-IDF_WIKI", rf_w2v_idf_wiki),
    ("RF+FT_WIKI", rf_ft_wiki),
    ("RF+FT-IDF_WIKI", rf_ft_idf_wiki),
]

###############################################################
print ("RESULTADOS - F1-Score, Acuracia, Precision e Recall") #
###############################################################

#------------------#
print ("F1-Score") #
#------------------#
from sklearn.model_selection import KFold
def benchmark_new_f1(model, X, y):
	scores = []
	kf = KFold(n_splits=10, random_state=66, shuffle=False)
	kf.get_n_splits(X, y)
	for train, test in kf.split(X, y):
		X_train, X_test = X[train], X[test]
		y_train, y_test = y[train], y[test]
		scores.append(f1_score(model.fit(X_train, y_train).predict(X_test), y_test, average = 'micro'))
		print (pd.DataFrame(scores)) # Guardar dados das 10 rodadas
	return np.mean(scores)

# ---------------------------------------------------------------------
table = []
t0 = time()
for name, model in all_models_svm:
	 print(name)
	 table.append({'model': name, 
				   'f1-score': benchmark_new_f1(model, X, y)})
	 print(table)

df_result_f1 = pd.DataFrame(table)
print(df_result_f1)
print("Resultados (SVM) - F1-Score - DONE in %0.3fs." % (time() - t0))

# -----------------------------
table = []
t0 = time()
for name, model in all_models_knn:
	 print(name)
	 table.append({'model': name, 
				   'f1-score': benchmark_new_f1(model, X, y)})
	 print(table)

df_result_f1 = pd.DataFrame(table)
print(df_result_f1)
print("Resultados (KNN) - F1-Score - DONE in %0.3fs." % (time() - t0))

# -----------------------------
table = []
t0 = time()
for name, model in all_models_dt:
	 print(name)
	 table.append({'model': name, 
				   'f1-score': benchmark_new_f1(model, X, y)})
	 print(table)

df_result_f1 = pd.DataFrame(table)
print(df_result_f1)
print("Resultados (Decision Tree) - F1-Score - DONE in %0.3fs." % (time() - t0))

# -----------------------------
table = []
t0 = time()
for name, model in all_models_rf:
	 print(name)
	 table.append({'model': name, 
				   'f1-score': benchmark_new_f1(model, X, y)})
	 print(table)

df_result_f1 = pd.DataFrame(table)
print(df_result_f1)
print("Resultados (Random Forest) - F1-Score - DONE in %0.3fs." % (time() - t0))

#------------------#
print ("Acuracia") #
#------------------#
from sklearn.model_selection import KFold
def benchmark_new(model, X, y):
    scores = []
    kf = KFold(n_splits=10, random_state=66, shuffle=False)
    kf.get_n_splits(X, y)
    for train, test in kf.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
        print (pd.DataFrame(scores)) # Guardar dados das 10 rodadas
    return np.mean(scores)

# ---------------------------------------------------------------------
table = []
t0 = time()
for name, model in all_models_svm:
     print(name)
     table.append({'model': name, 
                   'accuracy': benchmark_new(model, X, y)})
     print(table)

df_result = pd.DataFrame(table)
print(df_result)
print("Results (SVM) - Accuracy - DONE in %0.3fs." % (time() - t0))

# -----------------------------
table = []
t0 = time()
for name, model in all_models_knn:
     print(name)
     table.append({'model': name, 
                   'accuracy': benchmark_new(model, X, y)})
     print(table)

df_result = pd.DataFrame(table)
print(df_result)
print("Results (KNN) - Accuracy - DONE in %0.3fs." % (time() - t0))

# -----------------------------
table = []
t0 = time()
for name, model in all_models_dt:
     print(name)
     table.append({'model': name, 
                   'accuracy': benchmark_new(model, X, y)})
     print(table)

df_result = pd.DataFrame(table)
print(df_result)
print("Results (Decision Tree) - Accuracy - DONE in %0.3fs." % (time() - t0))

# -----------------------------
table = []
t0 = time()
for name, model in all_models_rf:
     print(name)
     table.append({'model': name, 
                   'accuracy': benchmark_new(model, X, y)})
     print(table)

df_result = pd.DataFrame(table)
print(df_result)
print("Results (Random Forest) - Accuracy - DONE in %0.3fs." % (time() - t0))

#-------------------#
print ("Precision") #
#-------------------#
from sklearn.model_selection import KFold
def benchmark_new_pr(model, X, y):
    scores = []
    kf = KFold(n_splits=10, random_state=66, shuffle=False)
    kf.get_n_splits(X, y)
    for train, test in kf.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(precision_score(model.fit(X_train, y_train).predict(X_test), y_test, average = 'micro'))
    return np.mean(scores)

table = []
t0 = time()
for name, model in all_models:
     print(name)
     table.append({'model': name, 
                   'precision': benchmark_new_pr(model, X, y)})
#df_result_pr = pd.DataFrame(sorted(table, reverse=True))
df_result_pr = pd.DataFrame(table)
print(df_result_pr)
print("Results - Precision - DONE in %0.3fs." % (time() - t0))

#----------------#
print ("Recall") #
#----------------#
from sklearn.model_selection import KFold
def benchmark_new_rc(model, X, y):
    scores = []
    kf = KFold(n_splits=10, random_state=66, shuffle=False)
    kf.get_n_splits(X, y)
    for train, test in kf.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(recall_score(model.fit(X_train, y_train).predict(X_test), y_test, average = 'micro'))
    return np.mean(scores)

table = []
t0 = time()
for name, model in all_models:
     print(name)
     table.append({'model': name, 
                   'recall..': benchmark_new_rc(model, X, y)})
#df_result_rc = pd.DataFrame(sorted(table, reverse=True))
df_result_rc = pd.DataFrame(table)
print(df_result_rc)
print("Results - Recall - DONE in %0.3fs." % (time() - t0))


#############################################################################
print ("Treinando e Predizendo - Melhores Classificadores - Mesma fonte") ###
#############################################################################

# Binarizes Labels
from sklearn.preprocessing import label_binarize

#Z5
name_labels = ['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews'] #Z5
Y = label_binarize(y, classes=['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews'])

n_classes = Y.shape[1]

# -------------------------
# Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=66) 

#SVM(RBF)+BoW      0,7873 (0,0008) - 1
#SVM(RBF)+FT+E2V   0,7846 (0,0004) - 2
#SVM(RBF)+W2V+E2V  0,7838 (0,0003) - 3

# Training
#svm_rbf_bow.fit(X_train, Y_train)
svm_rbf_ft_idf.fit(X_train, Y_train)
#svm_rbf_w2v_idf.fit(X_train, Y_train)

# Prediction
#predictions = svm_rbf_bow.predict(X_test)
predictions = svm_rbf_ft_idf.predict(X_test)
#predictions = svm_rbf_w2v_idf.predict(X_test)

# Reports
print ("Precision: %s" %precision_score(Y_test, predictions, average="micro"))
print ("Recall...: %s" %recall_score(Y_test, predictions, average="micro"))
print ("F1-Score.: %s" %f1_score(Y_test, predictions, average="micro"))
print ("Accuracy.: %s" %accuracy_score(Y_test, predictions))

print (classification_report(predictions,Y_test))

################
# SVM(RBF)+BoW #
################
# Len Test: 8507

# Precision: 0.8859878154917319
# Recall...: 0.7179969436934289
# F1-Score.: 0.793195247061879
# Accuracy.: 0.707299870694722

#               precision    recall  f1-score   support
#            0       0.71      0.90      0.79      1387
#            1       0.65      0.93      0.76      1146
#            2       0.80      0.97      0.88      1418
#            3       0.69      0.97      0.81      1178
#            4       0.74      0.73      0.73      1765
#    micro avg       0.72      0.89      0.79      6894
#    macro avg       0.72      0.90      0.79      6894
# weighted avg       0.72      0.89      0.79      6894
#  samples avg       0.72      0.71      0.71      6894
 
###################
# SVM(RBF)+FT+E2V #
###################
# Len Test: 8507

# Precision: 0.8673215273934698
# Recall...: 0.7369225343834489
# F1-Score.: 0.7968223705115983
# Accuracy.: 0.7266956623956742

#              precision    recall  f1-score   support
#           0       0.72      0.89      0.79      1417
#           1       0.66      0.90      0.76      1205
#           2       0.80      0.97      0.87      1416
#           3       0.72      0.95      0.82      1243
#           4       0.79      0.70      0.74      1947
#   micro avg       0.74      0.87      0.80      7228
#   macro avg       0.74      0.88      0.80      7228
#weighted avg       0.74      0.87      0.79      7228
# samples avg       0.74      0.73      0.73      7228

# Average precision score, micro-averaged over all classes: 0.87

####################
# SVM(RBF)+W2V+E2V #
####################
# Len Test: 8507

# Precision: 0.8685159500693481
# Recall...: 0.7360996826143176
# F1-Score.: 0.7968441814595659
# Accuracy.: 0.7268132126484071
# 
#               precision    recall  f1-score   support
#            0       0.72      0.88      0.79      1421
#            1       0.66      0.91      0.76      1191
#            2       0.79      0.97      0.87      1408
#            3       0.72      0.95      0.82      1247
#            4       0.79      0.71      0.75      1943
#    micro avg       0.74      0.87      0.80      7210
#    macro avg       0.74      0.88      0.80      7210
# weighted avg       0.74      0.87      0.80      7210
#  samples avg       0.74      0.73      0.73      7210

#----------------------------- #
#          Gráfico 1           #
# ---------------------------- #
# - Curve Precision - Recall - #
# ---------------------------- #

# Curva - Precision - Recall
y_score = svm_rbf_ft_idf.decision_function(X_test)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))


#------------------------ #
#       Gráfico 2
# ----------------------- #
# ------ Curve ROC ------ #
# ----------------------- #
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(Y_test.ravel(), y_score.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (Curve ROC)')
plt.legend(loc="lower right") # legenda
plt.show()

# ------------------------------------ #
#              Gráfico 3
# ------------------------------------ #
# ------ Curve ROC - Multiclass ------ #
# ------------------------------------ #
# Plot ROC curves for the multiclass problem

from itertools import cycle 
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='blue', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'brown', 'gray', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of category {0} (area = {1:0.2f})'
            #''.format(i, roc_auc[i]))
             ''.format(name_labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-labels')
plt.tight_layout()
#plt.legend(loc="lower right")
plt.legend(loc='lower right', ncol=2, borderaxespad=0, frameon=False)
plt.show()

#################################################################################
############################ TESTE COM OUTRAS FONTES ############################
#################################################################################

###################################################
print ("Importando as coleções de documentos...") #
###################################################
# Stemmed
X = pickle.load(open('data/reuters_X_pp.ipy', 'rb'))
y = pickle.load(open('data/reuters_y_pp.ipy', 'rb'))

X, y = np.array(X), np.array(y)

print ("Total de notícias curtas - Train: %s" % len(y))

# Test - Diversas fontes (reuters/inshorts)
###################################################
# Stemmed
Xt = pickle.load(open('data/reuters_X_Test_pp.ipy', 'rb'))
yt = pickle.load(open('data/reuters_y_Test_pp.ipy', 'rb'))

Xt, yt = np.array(Xt), np.array(yt)

print ("Total de notícias curtas - Test: %s" % len(yt))


# Training and Testing Sets
#################################################################
#Z5 - BR
name_labels = ['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews']
Y           = label_binarize(y,  classes=['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews'])
Yt          = label_binarize(yt, classes=['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews'])
n_classes = Y.shape[1]

X_train = X
Y_train = Y

X_test  = Xt
Y_test  = Yt

# Training
t0 = time()
svm_rbf_w2v_idf.fit(X_train, Y_train) # 419.247s.
print("Treinamento realizado em %0.3fs." % (time() - t0))

# Prediction
predictions = svm_rbf_w2v_idf.predict(X_test)

# Exportando...
pickle.dump(predictions, open('data/predictions_z5_svm_rbf_w2v_idf.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)

# Reports
print ("Precision: %s" %precision_score(Y_test, predictions, average="micro"))
print ("Recall...: %s" %recall_score(Y_test, predictions, average="micro"))
print ("F1-Score.: %s" %f1_score(Y_test, predictions, average="micro"))
print ("Accuracy.: %s" %accuracy_score(Y_test, predictions))

print (classification_report(predictions,Y_test))