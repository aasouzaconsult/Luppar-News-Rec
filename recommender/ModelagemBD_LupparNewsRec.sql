CREATE TABLE LupparNewsRec_PerfisAssinantes (
   id            SERIAL PRIMARY KEY,
   email         VARCHAR(50) NOT NULL,
   topicos       VARCHAR(255) NOT NULL,
   colecao       VARCHAR(50) NULL,
   classificador VARCHAR(50) NULL,
   representacao VARCHAR(50) NULL
);
-- DROP TABLE LupparNewsRec_PerfisAssinantes;

CREATE SEQUENCE assinantes_sequence
  start 1
  increment 1;
-- DROP SEQUENCE assinantes_sequence

-- Teste
insert into LupparNewsRec_PerfisAssinantes values (nextval('assinantes_sequence'), 'teste@teste.com', 'sportsnews, technologyNews', 'Z5News', 'SVM', 'FastText+E2V_IDF');

-- Visualizando
select * from LupparNewsRec_PerfisAssinantes