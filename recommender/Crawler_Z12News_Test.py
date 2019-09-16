# 0 technologyNews
# 1 aerospace-defense
# 2 autos
# 3 sportsnews
# 4 PersonalFinance
# 5 commoditiesNews
# 6 fundsNews
# 7 foreignexchangeNews
# 8 politicsNews
# 9 healthnews
# 10 environmentnews
# 11 brazil-news
from luppar.settings import BASE_DIR

def download_news_page_z12():
    import requests
    from bs4 import BeautifulSoup
    import json
    import pickle
    import os

    X = []
    y = []
    z = []
    dt = []
    lk = []

    def print_headlines(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('article')

        news_category = url.split('/')[5].split('?')[0]
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip() + ' - ' + headline.span.text)
                lk.append('https://www.reuters.com' + headline.a['href'])
            except:
                continue

    def print_headlines_autos(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('article')

        news_category = 'autos'
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip() + ' - ' + headline.span.text)
                lk.append('https://www.reuters.com' + headline.a['href'])
            except:
                continue

    def print_headlines_commo(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('article')

        news_category = 'commoditiesNews'
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip() + ' - ' + headline.span.text)
                lk.append('https://www.reuters.com' + headline.a['href'])
            except:
                continue

    def print_headlines_funds(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('article')

        news_category = 'fundsNews'
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip() + ' - ' + headline.span.text)
                lk.append('https://www.reuters.com' + headline.a['href'])
            except:
                continue

    def print_headlines_forei(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('article')

        news_category = 'foreignexchangeNews'
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip() + ' - ' + headline.span.text)
                lk.append('https://www.reuters.com' + headline.a['href'])
            except:
                continue

    def print_headlines_bra(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('article')

        news_category = 'brazil-news'
        for headline in headlines:
            try:
                X.append(headline.p.text)
                y.append(news_category)
                z.append(headline.p.text)
                dt.append(headline.h3.text.strip() + ' - ' + headline.span.text)
                lk.append('https://www.reuters.com' + headline.a['href'])
            except:
                continue

    # Ultimas notícias
    a = 1
    b = 2

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/technologyNews?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (AUG) - 1600
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/aerospace-defense?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (NOV) - 1000
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/autos-upclose?view=page&page=%d&pageSize=10" % (x) # Ano: 2017 (MAR) - 1000
       print(url)

       response = requests.get(url)
       print_headlines_autos(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/sportsnews?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (AUG) - 2800
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/PersonalFinance?view=page&page=%d&pageSize=10" % (x) # Ano: 2013 (SEP) - 1000
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/gca-commodities?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (AUG) - 2310
       print(url)

       response = requests.get(url)
       print_headlines_commo(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/fundsfundsnews?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (AUG) - 2310
       print(url)

       response = requests.get(url)
       print_headlines_funds(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/gca-foreignexchange?view=page&page=%d&pageSize=10" % (x) # Ano: 2011 (MAY) - 289*
       print(url)

       response = requests.get(url)
       print_headlines_forei(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/politicsNews?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (AUG) - 1670
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/healthnews?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (AUG) - 1080
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/environmentnews?view=page&page=%d&pageSize=10" % (x) # Ano: 2016 (AUG) - 1150
       print(url)

       response = requests.get(url)
       print_headlines(response.text)

    for x in range(a, b):
       url = "https://www.reuters.com/news/archive/brazil?view=page&page=%d&pageSize=10" % (x) # Ano: 2009 (JAN) - 1210*
       print(url)

       response = requests.get(url)
       print_headlines_bra(response.text)

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
                dt.append(
                    headline.find(itemprop='headline').text + ' - ' + headline.find(itemprop='dateModified')['content'])
                lk.append('https://inshorts.com' + headline.find('a')['href'])
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
                dt.append(
                    headline.find(itemprop='headline').text + ' - ' + headline.find(itemprop='dateModified')['content'])
                lk.append('https://inshorts.com' + headline.find('a')['href'])
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

        news_category = 'sportsnews'
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

    url = 'https://inshorts.com/en/read/sports'     # sportsnews

    response = requests.get(url)
    print_headlines_spo(response.text)

    #########################################
    # Crawler - Auto
    def print_headlines_aut(response_text):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines = soup.find_all('div', attrs={"class": "news-card"})

        news_category = 'autos'
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

    url = 'https://inshorts.com/en/read/automobile'     # autos

    response = requests.get(url)
    print_headlines_aut(response.text)

    #####################################
    # Importando e Pré-Processando

    def pesquise(lista, valor):
        for x, e in enumerate(lista):
            if e == valor:
                print(x)
                del X[x]
                del y[x]
                del z[x]
                del dt[x]
                del lk[x]
        return None

    pesquise(X, ' All quotes delayed a minimum of 15 minutes. See here for a complete list of exchanges and delays.')
    pesquise(X, "Reuters, the news and media division of Thomson Reuters, is the world’s largest international multimedia news provider reaching more than one billion people every day. Reuters provides trusted business, financial, national, and international news to professionals via Thomson Reuters desktops, the world's media organizations, and directly to consumers at Reuters.com and via Reuters TV.  Learn more about Thomson Reuters products:")

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

    data_dir = os.path.join(BASE_DIR, "data")

    pickle.dump(X, open(os.path.join(data_dir, 'reuters_X_Z12_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(y, open(os.path.join(data_dir, 'reuters_y_Z12_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(z, open(os.path.join(data_dir, 'reuters_z_Z12_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(dt, open(os.path.join(data_dir,'reuters_dt_Z12_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(lk, open(os.path.join(data_dir,'reuters_lk_Z12_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    download_news_page_z12()