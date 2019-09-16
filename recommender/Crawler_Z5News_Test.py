#################
## OBSERVAÇÕES ##
#################
# Fabiano, vamos começar usando apenas a coleção Z5News (Inglês)
# O arquivo de Treinamento, já pré-processado, se encontra em:
# data/reuters_X_pp_new.ipy
# data/reuters_y_pp_new.ipy
# data/reuters_z_pp_new.ipyimport pickle

# O script que faz a classificação é o: SRN_Monorotulo_Z5.py

# Esse aqui baixa notícias do site da Reuters e do site Inshorts
from luppar.settings import BASE_DIR


def download_news_page():
    ####################################################################
    import pickle
    import requests
    from bs4 import BeautifulSoup
    import os
    import json

    X = []
    y = []
    z = []  # esse criei pra manter a notícia inteira (retornar na tela)
    dt = []
    lk = []

    #########################################
    ############## REUTERS ##################
    #########################################

    def print_headlines(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('article')

        news_category = url.split('/')[5].split('?')[0]
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip()+' - '+headline.span.text)
                lk.append('https://www.reuters.com'+headline.a['href'])
            except:
                continue

    def print_headlines_fin(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('p')

        news_category = 'PersonalFinance'  # 21
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip()+' - '+headline.span.text)
                lk.append('https://www.reuters.com'+headline.a['href'])
            except:
                continue

    def print_headlines_bra(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('p')

        news_category = 'brazil-news'  # 21
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip()+' - '+headline.span.text)
                lk.append('https://www.reuters.com'+headline.a['href'])
            except:
                continue

    # Baixar as notícias das 2 primeiras páginas
    # ---------------------
    a = 1
    b = 3

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/sportsNews?view=page&page=%d&pageSize=10" % (x)
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/politicsNews?view=page&page=%d&pageSize=10" % (x)
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
      url = "https://www.reuters.com/news/archive/technologyNews?view=page&page=%d&pageSize=10" % (x)
      print(url)

      response = requests.get(url)
      print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/PersonalFinance?view=page&page=%d&pageSize=10" % (x)
       print(url)

       response = requests.get(url)
       print_headlines_fin(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/brazil?view=page&page=%d&pageSize=10" % (x)
       print(url)

       response = requests.get(url)
       print_headlines_bra(response.text)


    # -------------------------------------------------------------------------------------------------
    ##########################################
    ############## INSHORTS ##################
    ##########################################

    # Baixar as notícias...
    # ---------------------
    # Crawler - Politics News
    def print_headlines_pol(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('div', attrs={"class": "news-card"})

        news_category = 'politicsNews'
        for headline in headlines:
            try:
                X.append(headline.find(itemprop='articleBody').text)
                y.append(news_category)
                z.append(headline.find(itemprop='articleBody').text)
                dt.append(headline.find(itemprop='headline').text+' - '+headline.find(itemprop='dateModified')['content'])
                lk.append('https://inshorts.com'+headline.find('a')['href'])
            except:
                continue

    url = 'https://inshorts.com/en/read/politics'   # politicsNews

    response = requests.get(url)
    print_headlines_pol(response.text)

    #########################################
    # Crawler - Technology News
    def print_headlines_tec(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('div', attrs={"class": "news-card"})

        news_category = 'technologyNews'
        for headline in headlines:
            try:
                X.append(headline.find(itemprop='articleBody').text)
                y.append(news_category)
                z.append(headline.find(itemprop='articleBody').text)
                dt.append(headline.find(itemprop='headline').text+' - '+headline.find(itemprop='dateModified')['content'])
                lk.append('https://inshorts.com'+headline.find('a')['href'])
            except:
                continue


    url = 'https://inshorts.com/en/read/technology' # technologyNews

    response = requests.get(url)
    print_headlines_tec(response.text)

    #########################################
    # Crawler - Sports News
    def print_headlines_spo(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('div', attrs={"class": "news-card"})

        news_category = 'sportsNews'
        for headline in headlines:
            try:
                X.append(headline.find(itemprop='articleBody').text)
                y.append(news_category)
                z.append(headline.find(itemprop='articleBody').text)
                dt.append(
                    headline.find(itemprop='headline').text + ' - ' + headline.find(itemprop='dateModified')['content'])
                lk.append('https://inshorts.com' + headline.find('a')['href'])
            except:
                continue

    url = 'https://inshorts.com/en/read/sports'     # sportsNews

    response = requests.get(url)
    print_headlines_spo(response.text)

    ###################
    # Tratando a base #
    ###################

    # Apagando...
    def pesquise(lista, valor):
        for x, e in enumerate(lista):
            if e == valor:
                print(x)
                del X[x]
                del y[x]
                del z[x]
        return None

    pesquise(X, ' All quotes delayed a minimum of 15 minutes. See here for a complete list of exchanges and delays.')

    pesquise(X, "Reuters, the news and media division of Thomson Reuters, is the world’s largest international multimedia news provider reaching more than one billion people every day. Reuters provides trusted business, financial, national, and international news to professionals via Thomson Reuters desktops, the world's media organizations, and directly to consumers at Reuters.com and via Reuters TV.  Learn more about Thomson Reuters products:")

    # Salvando a coleção
    #pickle.dump(X, open('data/reuters_X_Test_new.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(y, open('data/reuters_y_Test_new.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(z, open('data/reuters_z_Test_new.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)


    ###################
    # Pré-Processando #
    ###################

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    from nltk.stem import WordNetLemmatizer
    from string import punctuation

    stopword = stopwords.words('english')

    snowball_stemmer = SnowballStemmer('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    def strip_punctuation(s):
        return ''.join(c for c in s if c not in punctuation)

    Xa = []
    Xa = X

    X = []

    for i in range(0,len(Xa)):
        text               = strip_punctuation(Xa[i]) # remove pontuacao
        text               = ''.join(c for c in text if not c.isdigit()) # remove numeros
        word_tokens        = nltk.word_tokenize(text.lower()) # tokenize
        removing_stopwords = [word for word in word_tokens if word not in stopword] # stopwords
        stemmed_word       = [snowball_stemmer.stem(word) for word in removing_stopwords] # stemmed
        X.append(stemmed_word)

    pesquise(X, ['a', 'chines', 'woman', 'got', 'secur', 'checkpoint', 'presid', 'donald', 'trump', 'maralago', 'resort', 'florida', 'carri', 'thumb', 'drive', 'code', 'malici', 'softwar', 'arrest', 'saturday', 'enter', 'restrict', 'properti', 'make', 'fals', 'statement', 'offici', 'accord', 'court', 'file', 'colett', 'luke'])

    # exportar

    data_dir = os.path.join(BASE_DIR,"data")

    pickle.dump(X, open(os.path.join(data_dir,'reuters_X_Z5_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(y, open(os.path.join(data_dir,'reuters_y_Z5_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(z, open(os.path.join(data_dir,'reuters_z_Z5_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(dt, open(os.path.join(data_dir, 'reuters_dt_Z5_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(lk, open(os.path.join(data_dir, 'reuters_lk_Z5_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    download_news_page()