# Luppar News-Rec
LupparNews-Rec: Um Recomendador Inteligente de Notícias

Softwares
---------
- Django
- Python 3.7 (principais bibliotecas: Numpy, NLTK, Scikit-Learn, Gensim, Matplotlib, BeautifulSoup)

Trabalho
------------
.pdf (será distribuida em breve)

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
Recursos do Luppar News-Rec
-----------
News Recommender System using Word Embeddings for study and research.

Link: 

Coleções de documentos
-----
- **Z5News** (Coleção em Inglês com 5 tópicos)
    - sportsNews, politicsNews, technologyNews, PersonalFinance e brazil-news 
- **Z5NewsBrasil** (Coleção em Portugues com 5 tópicos)
    - esporteNews, politicaNews, tecnologiaNews, financaPessoal e educacaonews
- **Z12News** (Coleção em Inglês com 12 tópicos)
    - sportsnews, politicsNews, technologyNews, PersonalFinance, brazil-news, aerospace-defense, autos, commoditiesNews, fundsNews, foreignexchangeNews, healthnews e environmentnews 

*Disponíveis em:* **/data**
- **Z5News** -> Arquivos de Treinamento: reuters_X_pp_new.ipy | reuters_y_pp_new.ipy | reuters_z_pp_new.ipy
- **Z5NewsBrasil** -> Arquivos de Treinamento: g1_X_pp_new.ipy | g1_y_pp_new.ipy | g1_z_pp_new.ipy
- **Z12News** -> Arquivos de Treinamento: reuters_X_Z12_pp.ipy | reuters_y_Z12_pp.ipy | reuters_z_Z12_pp.ipy
-----
- **Z5News** -> Arquivos de Teste: reuters_X_Test_pp_new_1d.ipy | reuters_y_Test_pp_new_1d.ipy | reuters_z_Test_pp_new_1d.ipy
- **Z5NewsBrasil** -> Arquivos de Teste: g1_X_test_pp_new_1d.ipy | g1_y_test_pp_new_1d.ipy | g1_z_test_pp_new_1d.ipy
- **Z12News** -> Arquivos de Teste: reuters_X_Z12_pp_Test_1d.ipy | reuters_y_Z12_pp_Test_1d.ipy | reuters_z_Z12_pp_Test_1d.ipy

Representações de Documentos
-------------
- **FastText + E2V-IDF** (Representação Embedding: FastText combinada com a abordagem E2V-IDF (ponderada por IDF))
- **Word2Vec + E2V-IDF** (Representação Embedding: Word2Vec combinada com a abordagem E2V-IDF (ponderada por IDF))
- **BoW** (Representação Bag-of-Words - saco de palavras)

Classificadores de Texto
-------------
- **SVM (RBF)** (Classificador SVM - Support Vector Machine com Kernel: RBF - Radial Basis Function)
- **Random Forest (RF)** (Classificador Random Forest - Floresta Aleatória)

Receba notícias por E-mail
-------------
Informe seu e-mail caso deseje receber notícias dos tópicos escolhidos por e-mail.

Botões
-------------
- **Recomendar** (Recomenda notícias, já armazenadas no Luppar News-Rec, conforme as seleções realizadas)
- **Baixar e Recomendar** (Baixa as últimas notícias, faz o pré-processamento, gera o modelo conforme seleção da representação e classificador informado e retorna as notícias recomendadas com base nos tópicos selecionados)
