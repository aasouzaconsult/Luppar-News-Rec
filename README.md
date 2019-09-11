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
- Z5News (Coleção em Inglês com 5 tópicos)
- Z5NewsBrasil (Coleção em Portugues com 5 tópicos)
- Z12News (Coleção em Inglês com 12 tópicos)

Disponíveis em: /data
- Z5News -> Arquivos de Treinamento: reuters_X_pp_new.ipy | reuters_y_pp_new.ipy | reuters_z_pp_new.ipy
- Z5NewsBrasil -> Arquivos de Treinamento: g1_X_pp_new.ipy | g1_y_pp_new.ipy | g1_z_pp_new.ipy
- Z12News -> Arquivos de Treinamento: reuters_X_Z12_pp.ipy | reuters_y_Z12_pp.ipy | reuters_z_Z12_pp.ipy
-----
- Z5News -> Arquivos de Teste: reuters_X_pp_new.ipy | reuters_y_pp_new.ipy | reuters_z_pp_new.ipy
- Z5NewsBrasil -> Arquivos de Teste: g1_X_pp_new.ipy | g1_y_pp_new.ipy | g1_z_pp_new.ipy
- Z12News -> Arquivos de Teste: reuters_X_Z12_pp.ipy | reuters_y_Z12_pp.ipy | reuters_z_Z12_pp.ipy

Representações de Documentos
-------------
- FastText + E2V-IDF ()
- BaseDemonstracao_Produto.xlsx

Scripts (.sql)
-------------
- AmbienteSQLServer.sql

Praticando em casa...
-------------
Usando o SQL Server ou qualquer outro SGBD, criar um banco de dados para Controle de Vendas, algumas das condições:
- Cada Venda tem que ter um Vendedor associado
- Cada Venda deve estar associada a um Cliente
- Cada Cliente terá um Nome, CPF, Telefone, Endereço completo (sugestão: Logradouro, CEP, Bairro, Cidade, Estado. Em campos separados)
- Cada Venda pode ter mais de um Produto (Item de Venda)
- Cada Produto deve ter um código, Descrição
- Para cada Item de Venda deve ser informado, o código e descrição do produto, quantidade vendida, o valor unitário e o Valor total

* Sugestão: alimentar com dados desde Janeiro de 2016, com clientes em diversos Bairros, Cidades e Estado. Usem a criatividade, exemplos (Ambiente_Exemplo_ControleDeVendas.sql e AmbienteSQLServer.sql) e experiência!

Gráficos (no Power BI ou qualquer outra ferramenta de Self-Service BI)
- Usem a criatividades
- Alguns exemplos: Total de Vendas por Ano, por Mês, por Estado (Mapa), Gráfico de Vendas e etc

--------------------------------------------------------------------------------------------------
2° Dia
-----------
ETL, Business Intelligence

Dicas
-----
- PowerBI (https://pessoalex.wordpress.com/bi-3/microsoft/power-bi/)
- Tableau (https://pessoalex.wordpress.com/bi-3/tableau/)

Scripts (.sql)
-------------
- AmbienteBI.sql
- Ambiente_Exemplo_ControleDeVendas.sql

Arquivos PowerBI (.pbix)
-------------
- ExemploPowerBI.pbix
- ExemploPowerBI_Dispersao_ENEM2014.pbix
- ExemploPowerBI_ConhecimentoESabedoria.pbix
- Exemplos online (https://docs.microsoft.com/pt-br/power-bi/sample-datasets)*
