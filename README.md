# LUPPAR

Sistema de Recuperação de Informação dotado de Análise de Contexto Local baseada em Modelo Semântico Distribucional

Url do Luppar:
--------------------------
http://luppar.com/

Requisitos:
--------------------------
- GIT (https://git-scm.com/download/win)
- MySQL (https://dev.mysql.com/downloads/mysql/)
- Requirements.txt
```
pip install -r requirements.txt
```

Conexão com o MySQL:
--------------------------
- 1. Criar um Schema (comando abaixo:)
```
CREATE SCHEMA `luppar` DEFAULT CHARACTER SET utf8 ;
```

- 2. Criar a conexão com o Banco de Dados

Na pasta: \test
- Copiar o arquivo env.example e criar como .env. Configurar o BD (exemplo)
```
DEBUG=True
SECRET_KEY=secret
DATABASE_NAME=luppar
DATABASE_URL=localhost
DATABASE_USER=root
DATABASE_PASS=xxxxxxxx@yyy
DATABASE_PORT=3306
```

OU

Conexão com SQLLite(local)
--------------------------
Alterar o arquivo: \luppar\settings.py no trecho abaixo:
```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3')
        #'ENGINE': 'django.db.backends.mysql',
        #'NAME': config('DATABASE_NAME'),
        #'USER': config('DATABASE_USER'),'PASSWORD': config('DATABASE_PASS'),'HOST': config('DATABASE_URL'),
        #'PORT': config('DATABASE_PORT'),
    }
}
```
Montando o ambiente (migrando as tabelas):
--------------------------
```
python manage.py migrate
```

Importar dados
```
python manage.py loaddata data/datadump_basic.json
```

Subir o servidor:
--------------------------
```
python manage.py runserver
```

Caminho padrão do servidor:
--------------------------
http://localhost:8000/