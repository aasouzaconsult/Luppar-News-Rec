from time import time

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


from engine.util import load_object, E2V_IDF
from django.conf import  settings

from recommender.Crawler_Z12News_Test import download_news_page_z12
from recommender.Crawler_Z5NewsBrasil_Test import download_news_page_z5br
from recommender.Crawler_Z5News_Test import download_news_page


class DocRecom(object):
    def __init__(self, doc, label, dt, lk):
        self.doc = doc
        self.label = label
        self.dt = dt
        self.lk = lk


def search_news(name, name_labels, source, down=False, metrics=False):

    svm_rbf_ft_idf = load_object(settings.BASE_DIR+"/model/"+ name +".ipy")

    source_site = ''
    al_classes = []
    if(source == 'Z12'):
        source_site = 'reuters'
        all_classes = ['technologyNews', 'aerospace-defense', 'autos', 'sportsnews', 'PersonalFinance',
                       'commoditiesNews', 'fundsNews', 'foreignexchangeNews', 'politicsNews', 'healthnews',
                       'environmentnews', 'brazil-news']
        if down:
            download_news_page_z12()
    elif (source == 'Z5'):
        source_site = 'reuters'
        all_classes = ['technologyNews', 'PersonalFinance', 'sportsNews', 'brazil-news', 'politicsNews']
        if down:
            download_news_page()

    elif (source == 'Z5BR'):
        source_site = 'g1'
        all_classes = ['esporteNews', 'politicaNews', 'tecnologiaNews', 'financaPessoal', 'educacaonews']
        if down:
            download_news_page_z5br()


    ############################         TESTE - 1 DIA         ############################
    # Test - Diversas fontes (Reuters e Inshorts) - 1 dia
    #####################################################
    # Stemmed
    Xt = load_object(settings.BASE_DIR+'/data/'+source_site+'_X_'+source+'_pp_Test_1d.ipy')
    yt = load_object(settings.BASE_DIR+'/data/'+source_site+'_y_'+source+'_pp_Test_1d.ipy')
    zt = load_object(settings.BASE_DIR+'/data/'+source_site+'_z_'+source+'_pp_Test_1d.ipy')
    dtt = load_object(settings.BASE_DIR+'/data/'+source_site+'_dt_'+source+'_pp_Test_1d.ipy')
    lkt = load_object(settings.BASE_DIR+'/data/'+source_site+'_lk_'+source+'_pp_Test_1d.ipy')


    Yt = label_binarize(yt, classes=all_classes)

    n_classes = len(all_classes)

    X_test = Xt
    Z_test = zt
    Y_test = Yt

    print("Predição...")
    predictions = svm_rbf_ft_idf.predict(X_test)

    docsRetornados = []

    # Salvando os documentos e labels previstos...
    for j in range(len(predictions)):
        for i in range(n_classes):
            if predictions[j][i] == 1:
                idx = i
                if all_classes[idx] in name_labels:
                    try:
                        dc = DocRecom(Z_test[j],all_classes[idx],dtt[j],lkt[j])
                        docsRetornados.append(dc)
                    except:
                        continue

    metrics_result = ''

    if metrics:
        metrics_result = classification_report(Y_test, predictions, target_names=all_classes)

    return docsRetornados, metrics_result


