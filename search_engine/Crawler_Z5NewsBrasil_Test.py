# Base de Teste - Várias fontes (UOL):
# - Folha de S. Paulo
# - Estadão Conteúdo - Política
# - UOL Notícias - Política
# - UOL Notícias - Internacional
# - UOL Notícias - Educação
# - BBC News Brasil
# - BBC News Brasil - Internacional
# - Globo Esporte 
# - Lance
# - Canaltech
# - MacMagazine
# - Olhar Digital
# - Gizmodo
# - Reuters Brasil
# - Bloomberg
# - Brasil Escola
# - Agência Brasil
from luppar.settings import BASE_DIR


def download_news_page_z5br():

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

    def print_headlines(response_text, news_category):
        soup = BeautifulSoup(response_text, 'lxml')
        headlines  = soup.find_all('div', attrs={"class": "thumbnail-standard-wrapper"})

        for headline in headlines:
            try:
                X.append(headline.a.p.text)
                y.append(news_category)
                z.append(headline.a.p.text)
                dt.append(headline.a.h3.text + ' - ' + headline.a.time.text)
                lk.append(headline.a['href'])
            except:
                continue


    # baixando...
    #------------------------------------------------------
    url = "https://noticias.uol.com.br/politica/"
    response = requests.get(url)
    news_category = "politicaNews"
    print_headlines(response.text,news_category)

    #------------------------------------------------------
    url = "https://esporte.uol.com.br/futebol/ultimas"
    news_category = "esporteNews"
    response = requests.get(url)
    print_headlines(response.text,news_category)

    #------------------------------------------------------
    url = "https://noticias.uol.com.br/tecnologia/ultimas"
    news_category = "tecnologiaNews"
    response = requests.get(url)
    print_headlines(response.text,news_category)

    #------------------------------------------------------
    url = "https://economia.uol.com.br/noticias//index.htm"
    news_category = "financaPessoal"
    response = requests.get(url)
    print_headlines(response.text,news_category)

    #------------------------------------------------------
    url = "https://educacao.uol.com.br/ultimas"
    news_category = "educacaonews"
    response = requests.get(url)
    print_headlines(response.text,news_category)

    ###############################################################################################################
    # Importação e Pré-Processando
    ###############################################################################################################
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    from nltk.stem import WordNetLemmatizer
    from string import punctuation

    stopword = stopwords.words('portuguese')
    stopword.append('g1')
    stopword.append('globo.com')
    stopword.append('“')
    stopword.append('”')
    stopword.append('€')
    stopword.append('ª')
    stopword.append('–')
    stopword.append('º')
    stopword.append('’')
    stopword.append(']')
    stopword.append('[')
    stopword.append('—')

    stopword.append('é')
    stopword.append('a')
    stopword.append('b')
    stopword.append('c')
    stopword.append('d')
    stopword.append('e')
    stopword.append('f')
    stopword.append('g')
    stopword.append('h')
    stopword.append('i')
    stopword.append('j')
    stopword.append('l')
    stopword.append('m')
    stopword.append('n')
    stopword.append('o')
    stopword.append('p')
    stopword.append('q')
    stopword.append('r')
    stopword.append('s')
    stopword.append('t')
    stopword.append('u')
    stopword.append('v')
    stopword.append('x')
    stopword.append('z')
    stopword.append('w')
    stopword.append('y')

    stopword.append('reinaldo')
    stopword.append('azevedo')
    stopword.append('uol')
    stopword.append('notícias')
    stopword.append('política')
    stopword.append('estadão')
    stopword.append('conteúdo')
    stopword.append('tales')
    stopword.append('faria')
    stopword.append('bbc')
    stopword.append('news')
    stopword.append('brasil')
    stopword.append('hospício')
    stopword.append('sigilo')
    stopword.append('macmagazine')
    stopword.append('canaltech')
    stopword.append('adrenaline')
    stopword.append('olhar')
    stopword.append('digital')
    stopword.append('gizmodo')
    stopword.append('gesner')
    stopword.append('oliveira')
    stopword.append('reuters')
    stopword.append('folha')
    stopword.append('blog')
    stopword.append('joão')
    stopword.append('antônio')
    stopword.append('motta')
    stopword.append('todos')
    stopword.append('bordo')
    stopword.append('escola')
    stopword.append('agência')
    stopword.append('cult')
    stopword.append('gv')
    stopword.append('educação')
    stopword.append('kids')
    stopword.append('descomplique')
    stopword.append('jc')
    stopword.append('online')
    stopword.append('angela')
    stopword.append('moon')
    stopword.append('plano')
    stopword.append('carreira')
    stopword.append('luciano')
    stopword.append('costa')
    stopword.append('marcelo')
    stopword.append('rochabrun')
    stopword.append('andressa')
    stopword.append('pellanda')
    stopword.append('sheila')
    stopword.append('dang')
    stopword.append('akanksha')
    stopword.append('rana')
    stopword.append('hilary')
    stopword.append('russ')
    stopword.append('ayhan')
    stopword.append('uyanik')
    stopword.append('paul')
    stopword.append('lienert')
    stopword.append('ankit')
    stopword.append('ajmera')
    stopword.append('jeffrey')
    stopword.append('dastin')
    stopword.append('heather')
    stopword.append('somerville')
    stopword.append('cotidiano')
    stopword.append('tom')
    stopword.append('wilson')
    stopword.append('Dan')
    stopword.append('williams')
    stopword.append('margaryta')
    stopword.append('chornokondratenko')
    stopword.append('makiko')
    stopword.append('yamazaki')
    stopword.append('bharath')
    stopword.append('manjeshr')
    stopword.append('aparajita')
    stopword.append('saxena')
    stopword.append('sam')
    stopword.append('nussey')
    stopword.append('ficadica')
    stopword.append('nick')
    stopword.append('carey')
    stopword.append('biologianet')
    stopword.append('kate')
    stopword.append('holton')
    stopword.append('joshua')
    stopword.append('franklin')
    stopword.append('paresh')
    stopword.append('dave')
    stopword.append('uday')
    stopword.append('sampath')
    stopword.append('kumar')

    snowball_stemmer = SnowballStemmer('portuguese')

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

    pickle.dump(X, open(os.path.join(data_dir, 'g1_X_Z5BR_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(y, open(os.path.join(data_dir, 'g1_y_Z5BR_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(z, open(os.path.join(data_dir, 'g1_z_Z5BR_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(dt, open(os.path.join(data_dir, 'g1_dt_Z5BR_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(lk, open(os.path.join(data_dir, 'g1_lk_Z5BR_pp_Test_1d.ipy'), 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    download_news_page_z5br()