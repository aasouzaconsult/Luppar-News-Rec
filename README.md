# *Luppar News-Rec*
Luppar News-Rec: Um Recomendador Inteligente de Notícias

*O **Luppar News-Rec** é um SRN composto por algoritmos clássicos de classificação que trabalham em conjunto com representações de documentos para solucionar o problema de classificação de notícias de forma a trazer os documentos(notícias) que atendam a necessidade do usuário. A metodologia que segue esse trabalho divide os esforços da implementação do SRN em três etapas: Subsistema de Captura, Pré-Processamento e Armazenamento, Subsistema de Classificação de Notícias e Subsistema de Aquisição de Perfil de Assinantes e Distribuição.*

Softwares
---------
- <a href='https://www.djangoproject.com/'>Django</a>
- Python 3.7 (principais bibliotecas: <a href='https://numpy.org/'>Numpy</a>, 
                                      <a href='https://www.nltk.org/'>NLTK</a>,
                                      <a href='https://scikit-learn.org/stable/'>Scikit-Learn</a>,
                                      <a href='https://radimrehurek.com/gensim/'>Gensim</a>,
                                      <a href='https://matplotlib.org/'>Matplotlib</a>,
                                      <a href='https://www.crummy.com/software/BeautifulSoup/bs4/doc/'>BeautifulSoup</a>)



Trabalho
------------
~~LupparNews-Rec.pdf (será distribuida em breve)~~

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
Recursos do *Luppar News-Rec*
-----------
*News Recommender System using Word Embeddings for study and research.*

Dísponivel em: http://luppar.com/recommender

Coleções de documentos
-----
- **Z5News** (Coleção em Inglês com 5 tópicos)
    - sportsNews, politicsNews, technologyNews, PersonalFinance e brazil-news 
- **Z5NewsBrasil** (Coleção em Portugues com 5 tópicos)
    - esporteNews, politicaNews, tecnologiaNews, financaPessoal e educacaonews
- **Z12News** (Coleção em Inglês com 12 tópicos)
    - sportsnews, politicsNews, technologyNews, PersonalFinance, brazil-news, aerospace-defense, autos, commoditiesNews, fundsNews, foreignexchangeNews, healthnews e environmentnews 

*Disponíveis em:* **/data**
- **Z5News** -> Arquivos de Treinamento com 42.532 notícias do site <a href='https://www.reuters.com/'>reuters</a>.
    - reuters_X_pp_new.ipy *(Notícias - pré-processadas)*
    - reuters_y_pp_new.ipy *(Tópicos)*
    - reuters_z_pp_new.ipy *(Notícias - completas)*
- **Z5NewsBrasil** -> Arquivos de Treinamento com 24.177 notícias do site <a href='https://g1.globo.com/'>g1 news</a>. 
    - g1_X_pp_new.ipy *(Notícias - pré-processadas)*
    - g1_y_pp_new.ipy *(Tópicos)*
    - g1_z_pp_new.ipy *(Notícias - completas)*
- **Z12News** -> Arquivos de Treinamento com 41.849 notícias do site <a href='https://www.reuters.com/'>reuters</a>.
    - reuters_X_Z12_pp.ipy *(Notícias - pré-processadas)*
    - reuters_y_Z12_pp.ipy *(Tópicos)*
    - reuters_z_Z12_pp.ipy *(Notícias - completas)*
-----
- **Z5News** -> Arquivos de Teste com 187 notícias dos sites: <a href='https://www.reuters.com/'>reuters</a> e <a href='https://inshorts.com/en/read'>inshorts</a>.
    - reuters_X_Test_pp_new_1d.ipy *(Notícias de teste - pré-processadas)*
    - reuters_y_Test_pp_new_1d.ipy *(Tópicos)*
    - reuters_z_Test_pp_new_1d.ipy *(Notícias de teste - completas)*
- **Z5NewsBrasil** -> Arquivos de Teste com 189 notícias do site: <a href='https://noticias.uol.com.br/'>uol notícias (agregador)</a>.
    - g1_X_test_pp_new_1d.ipy *(Notícias de teste - pré-processadas)*
    - g1_y_test_pp_new_1d.ipy *(Tópicos)*
    - g1_z_test_pp_new_1d.ipy *(Notícias de teste - completas)*
- **Z12News** -> Arquivos de Teste com 232 notícias dos sites: <a href='https://www.reuters.com/'>reuters</a> e <a href='https://inshorts.com/en/read'>inshorts</a>.
    - reuters_X_Z12_pp_Test_1d.ipy *(Notícias de teste - pré-processadas)*
    - reuters_y_Z12_pp_Test_1d.ipy *(Tópicos)*
    - reuters_z_Z12_pp_Test_1d.ipy *(Notícias de teste - completas)*

Representações de Documentos
-------------
- **FastText + E2V-IDF** (Representação *Embedding*: FastText combinada com a abordagem E2V-IDF (ponderada por IDF))
- **Word2Vec + E2V-IDF** (Representação *Embedding*: Word2Vec combinada com a abordagem E2V-IDF (ponderada por IDF))
- **BoW** (Representação *Bag-of-Words* - saco de palavras)

Classificadores de Texto
-------------
- **SVM (RBF)** (Classificador SVM - *Support Vector Machine* com *Kernel*: RBF - *Radial Basis Function*)
- **Random Forest (RF)** (Classificador *Random Forest* - Floresta Aleatória)

Receba notícias por E-mail (Crie seu perfil de Assinante)
-------------
Informe seu e-mail caso deseje receber notícias diariamente (1x) via e-mail.
- Para salvar seu perfil de assinante:
    - informe seu e-mail, escolha os tópicos de sua preferência e clique no botão: **Recomendar**.

Métricas
-------------
Informe **Sim** caso deseje que as métricas da combinação escolhida sejam expostas na tela.

Botões
-------------
- **Recomendar** (Recomenda notícias, já armazenadas no Luppar News-Rec, conforme as seleções realizadas)
- **Baixar e Recomendar** (Baixa as últimas notícias, faz o pré-processamento, gera o modelo conforme seleção da representação e classificador informado e retorna as notícias recomendadas com base nos tópicos selecionados)
    - **Subsistema 1** - Baixar (*web crawler*), Pré-Processar (Remoção de Pontuação, Numeração, *StopWords*, aplicado *Stemming*) e armazenamento das notícias;
    - **Subsistema 2** - Classificar de forma automática em tópicos, fazendo uso de *Machine learning*, as notícias armazenadas na etapa anterior (*Subsistema 1*), utilizando combinações entre representação de documentos e classificadores de texto;
    - **Subsistema 3** - Recomenda as notícias classificadas em tópicos na etapa anterior (*Subsistema 2*) com base nas preferências dos usuários (utilizando abordagem baseado em conteúdo (subscrição em itens)).

Observações
-------------
- **Vetores *embeddings* para coleções Z5News e Z12News** (Se encontram na pasta: **\data\emb**)
- **Modelos gerados para coleções Z5News, Z5NewsBrasil e Z12News** (Se encontram na pasta: **\data\pred**)
- **Página** (Se encontram na pasta: **\templates**)
- **Implementação** (Se encontram na pasta: ~~**\engine** será distribuida em breve~~)

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
Dúvidas ou Sugestões:
-----------
<a href="mailto:aasouzaconsult@gmail.com">Via E-mail</a>
