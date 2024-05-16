# **<p align="center">Análise de Crédito para Fintech – PoD Bank</p>**

<p align="center">
  <img src="https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/imagem_capa_01.jpeg" alt="imagem_capa">
</p>

<br><br>
## **Sumário**
1. [Resumo Executivo](#1--resumo-executivo)
2. [Introdução](#2-introdução)
3. [Entendimento do Problema](#3-entendimento-do-problema)
4. [Entendimento do Negócio](#4-entendimento-do-negócio)
5. [Entendimento dos Dados](#5-entendimento-dos-dados)
6. [Modelo Baseline](#6-modelo-baseline)
7. [Feature Engineering](#7-feature-engineering)
8. [Preparação dos Dados (DataPrep)](#8-preparação-dos-dados-dataprep)
9. [Treinamento do Modelo (Regressão Logística)](#9-treinamento-do-modelo-regressão-logística)
10. [Treinamento do Modelo (Desafiante)](#10-treinamento-do-modelo-desafiante)
11. [Conclusão](#11-conclusão)
12. [Próximos Passos](#12-próximos-passos)

<br><br>
## **1.  Resumo Executivo**
O projeto se baseia em modelagem para concessão de crédito para uma fintech. A PoD Bank, Fintech que concede crédito para população com pouca informação de crédito, precisa de modelos (ML) para concessão de crédito. Este projeto teve como objetivo desenvolver um modelo de crédito eficaz e robusto, capaz de prever com precisão a probabilidade de inadimplência de um cliente. Para isso, foramrealizadas as seguintes etapas:
- **Feature Engineering:** Criação de novas variáveis a partir das variáveis existentes para enriquecer o treinamento dos modelos.
- **Preparação dos Dados:** Realização de uma análise detalhada das variáveis, tratamento dos dados ausentes e preparação as variáveis para o treinamento dos modelos, utilizando técnicas como padronização e codificação de variáveis categóricas. Além de selecionar as variáveis mais importantes para o treinamento do modelo.
-	**Treinamento do Modelo (Regressão Logística):** Utilizamos a Regressão Logística como primeiro modelo, devido à sua simplicidade e facilidade de interpretação. Obtivemos resultados satisfatórios, com métricas como AUC-ROC, Gini e KS superiores ao modelo baseline.
-	**Treinamento do Modelo (Desafiante):** Exploramos algoritmos mais avançados, como Gradient Boosting, XGBoost e LightGBM, buscando melhorar ainda mais o desempenho do modelo. Utilizamos técnicas de tunagem de hiperparâmetros com Grid Search e Optuna, priorizando as métricas de desempenho mais relevantes.
-	**Escolha do Melhor Modelo:** Com base nas métricas de desempenho e na análise comparativa, foi escolhido o modelo treinado com XGBoost utilizando Optuna como o modelo desafiante, além do modelo de Regressão Logística, devido às suas métricas superiores e ao menor tempo de processamento.

<br><br>
## **2. Introdução**
A concessão de crédito envolve riscos significativos para as empresas fornecedoras, especialmente quando há inadimplência por parte dos clientes. Aqui estão alguns problemas e riscos associados à inadimplência na concessão de crédito:

- **Perda Financeira:** A inadimplência pode resultar em perda financeira direta para a empresa, uma vez que os valores devidos podem não ser recuperados. Isso impacta diretamente a rentabilidade e a saúde financeira da empresa.
- **Redução da Liquidez:** A inadimplência pode afetar a liquidez da empresa, pois os fundos que deveriam ser recebidos podem não estar disponíveis para cobrir despesas operacionais, pagamento de fornecedores ou investimentos.
- **Aumento de Custos de Recuperação:** Recuperar valores de clientes inadimplentes pode exigir esforços significativos e custos adicionais. Isso inclui custos legais, custos de cobrança e tempo dedicado à gestão de recuperação.
- **Avaliação de Crédito Inadequada:** Uma avaliação inadequada do crédito, incluindo a falta de análise detalhada do histórico financeiro dos clientes, capacidade de pagamento e comportamento de crédito passado, pode levar a concessões de crédito para clientes de alto risco.
- **Fraude:** A concessão de crédito sem uma avaliação adequada pode abrir espaço para atividades fraudulentas. Os clientes podem fornecer informações falsas ou intencionalmente buscar crédito com a intenção de não honrar os compromissos.

A modelagem de crédito oferece uma solução para esses desafios. Ao utilizar técnicas estatísticas e algoritmos de machine learning, a modelagem de crédito pode prever o risco de inadimplência de um cliente com base em seu perfil financeiro e comportamental. Isso permite que as empresas identifiquem e avaliem com mais precisão os clientes de alto risco, evitando concessões de crédito inadequadas.

Além disso, a modelagem de crédito pode ajudar as empresas a desenvolverem estratégias de gerenciamento de risco mais eficazes. Isso inclui a definição de limites de crédito adequados, a segmentação de clientes com base em seu perfil de risco e a implementação de políticas de cobrança mais assertivas.

<br><br>
## **3. Entendimento do Problema**
A PoD Bank, uma startup do segmento financeiro que concede crédito para população com pouca informação de crédito, ganhou mercado e maturidade. Com isso, começou a sentir necessidade de um modelo de crédito para amparar suas decisões e deixar de depender apenas do BI. Decidiu, então, montar uma área de Planejamento e Modelagem de Crédito focada em modelos estatísticos.

No entanto, a Head de Crédito, está começando a ficar preocupada pois, embora o crescimento da carteira de clientes PoD Bank seja expressivo, ela está vendo a inadimplência do mercado subir. Por essa razão ela solicitou à área de Planejamento e Modelagem da PoD Bank um **Modelo de Application**, que tenha capacidade de gerar um score de risco para contratação de produtos de crédito.

O problema apresentado traz o desafio de prever a capacidade de um cliente para reembolsar um empréstimo. Neste cenário, a empresa PoD Bank se esforça para expandir o acesso ao crédito para uma clientela com histórico de crédito insuficiente ou inexistente. Utilizando uma variedade de fontes de dados, incluindo registros de transações bancárias, histórico de pagamentos, e informações socioeconômicas, o objetivo é construir um modelo preditivo que possa avaliar a probabilidade de inadimplência de um cliente.

Este problema é crucial porque ajuda a empresa a minimizar os riscos de crédito ao mesmo tempo em que possibilita o acesso a empréstimos para indivíduos que tradicionalmente não teriam essa oportunidade. Portanto, trata-se de um equilíbrio entre responsabilidade financeira e inclusão social.

As principais métricas que serão levadas em conta como parâmetro para a escolha do melhor modelo são AUC-ROC, Gini e KS, além de apresentar é claro uma boa ordenação das faixas de score.

<br><br>
## **4. Entendimento do Negócio**
Com base na análise exploratória dos dados fornecidos pela empresa (arquivo [**01_Entendimento_dos_Dados**](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/01_Entendimento_dos_Dados.ipynb)), foi possível gerar alguns insights valiosos sobre o perfil do cliente, padrões de pagamento e riscos associados à carteira de crédito, o que pode ajudar as instituições financeiras a tomar decisões mais informadas e eficazes. Analisar o cenário atual da carteira também é essencial para contextualizar os resultados do modelo de previsão de inadimplência e garantir que ele seja eficaz e relevante para as condições atuais do mercado e do cliente.
<br><br>
**Alguns dados sobre a carteira atual da empresa:**
- **Renda Média:** $ 168.557
- **Média do Valor de Empréstimo:** $ 599.496
- **Média de Idade:** 44 anos
- **Média de Membros na Família:** 2 pessoas
<br><br>

A partir da plotagem dos gráficos de colunas podemos obter mais algumas informações sobre o perfil dos nossos clientes como, por exemplo:
- **Gênero:** Podemos verificar que mais de 65% dos clientes são do gênero feminino, quase o dobro de clientes.
- **Estado Civil**: 64% dos clientes são casados.
- **Filhos:** A grande maioria dos clientes não possuem filhos.
- **Escolaridade:** O Ensino Médio é o nível escolar mais comum entre os clientes, com quase 71%, bem acima dos demais níveis.
- **Profissão:** Mais da metade dos clientes trabalham e a profissão mais comum entre eles é a de operário.

<br><br>
![grafico_barras](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/grafico_barras.png)

<br><br>
**Cenário atual da carteira:**
- **Quantidade de empréstimos:** 215.257
- **Quantidade de Cash loans:** 194.663
- **Quantidade de Revolving loans:** 20.594
- **Valor médio dos empréstimos:** 599.496
- **Valor médio por Cash loans:** 628.525
- **Valor médio por Revolving loans:** 325.106

<br><br>
![distribuicao_var_target](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/distribuicao_var_target.png)

<br><br>
![distribuicao_tipo_contrato](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/distribuicao_tipo_contrato.png)

<br><br>
![comparacao_taxa_maus](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/comparacao_taxa_maus.png)

- É possível verificar que a taxa de inadimplentes é baixa, menos de 10% da carteira.
- O tipo de empréstimo “Cash loans” é o mais solicitado pelos clientes, mais de 90% da carteira é de “Cash loans”.
- Já a taxa de maus por produto não foge muito da taxa de maus da carteira como um todo, com “Revolving loans” tendo uma taxa menor de maus que “Cash loans”.

<br><br>
## **5. Entendimento dos Dados**
O processo de entendimento dos dados é fundamental no treinamento de modelos de machine learning por várias razões cruciais. Primeiramente, ele nos permite obter insights sobre a natureza e a qualidade dos dados que estão sendo utilizados, identificando potenciais problemas como dados ausentes, valores discrepantes ou desbalanceamento de classes. Isso possibilita a escolha adequada das técnicas de pré-processamento necessárias para limpar e preparar os dados de forma apropriada para o treinamento do modelo. Além disso, compreender os dados ajuda na seleção das features mais relevantes e informativas para alimentar o modelo, o que pode melhorar significativamente sua capacidade de generalização e desempenho preditivo.

Os dados utilizados para a análise são provenientes de fontes internas e externas, conferindo assim uma perspectiva ampla e detalhada acerca do perfil dos solicitantes. Os dados internos abrangem informações cadastrais e transacionais.

Para resolver o problema de negócio a empresa reuniu os dados que estavam disponíveis e disponibilizou na pasta [**database**](https://drive.google.com/drive/folders/113P7yd2mVVpuNKuPEE-zw41_pxCfCCQS?usp=drive_link), no formato de arquivo CSV, para o treinamento e avaliação dos modelos. Abaixo segue uma breve descrição do conteúdo de cada arquivo disponibilizado:

- **application_train.csv:** Estes são os principais dados de treino com informações sobre cada solicitação de empréstimo na PoD Bank. Cada empréstimo tem sua própria linha e é identificado pela variável **SK_ID_CURR**. Os dados de aplicação de treinamento já vêm com a variável **TARGET**, indicando 0 (o empréstimo foi pago) ou 1 (o empréstimo não foi pago - inadimplência)
- **application_test.csv:** São os dados que usaremos para fazer a escoragem dos dados no modelo e simular um ambiente real. Essa tabela não possui a variável **TARGET**.
- **previous_application.csv:** Contém informações sobre aplicações de empréstimo anteriores de um cliente na PoD Bank.
- **POS_CASH_balance.csv:** Informações sobre o histórico de pagamentos de POS (Point of Sale) ou empréstimos em dinheiro na PoD Bank.
- **installments_payments.csv:** Detalha o histórico de pagamentos de empréstimos anteriores na PoD Bank.
- **credit_card_balance.csv:** Informações mensais sobre saldos de cartões de crédito do cliente na PoD Bank.
- **bureau.csv:** Fornece dados de crédito de outras instituições financeiras.
- **bureau_balance.csv:** Informações mensais sobre créditos anteriores do cliente em outras instituições financeiras.

Para obter os metadados, contendo a descrição de cada variável de cada arquivo, consulte o arquivo **HomeCredit_columns_description.csv**.

<br><br>
![schema](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/schema.jpeg)

<br><br>
Ao fazer a análise exploratória dos dados (arquivo [**01_Entendimento_dos_Dados**](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/01_Entendimento_dos_Dados.ipynb)), foi possível observar a ausência de datas na base principal **application_train.csv**, sendo assim sabemos que não será possível trabalhar com safras neste projeto. Vimos também que a base já possui a variável target, então não será necessário fazer a construção do target para este projeto.

De modo geral, não há nenhuma categoria que esteja muito relacionada com o target, a distribuição é bem próxima. Na variável NAME_INCOME_TYPE até existem categorias com predominância de inadimplentes ou adimplentes, mas essas categorias não são representativas, pois possuem menos de 1% de registros, então não teriam um bom poder preditivo para o modelo.

<br><br>
## **6. Modelo Baseline**
Primeiramente foi criado um modelo básico, sem muitas técnicas e métodos, utilizando somente a base **application_train.csv** que já está disponível, a partir do treinamento de alguns algoritmos, para saber como eles se saem em questão de desempenho, para que a partir daí seja possível melhorar essa performance, seja criando features, seja utilizando outras técnicas, tratamentos e etc.

Fazer um modelo baseline é essencial ao treinar um modelo de machine learning, pois fornece uma referência inicial para entender a complexidade do problema, estabelece um benchmark para comparação com modelos mais avançados, identifica problemas nos dados ou formulação do problema e facilita a comunicação com stakeholders ao oferecer uma referência compreensível para discutir desempenho e resultados esperados do modelo.

O código desse modelo baseline pode ser encontrado no arquivo [**01_Modelo_Baseline**](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_baseline/01_Modelo_Baseline.ipynb).
<br><br>
### **6.1 Ambiente de Desenvolvimento:**
Utilizamos o ambiente Python do Google Colab para a manipulação dos dados e treinamento do modelo baseline.
<br><br>
### **6.2 Pacotes e bibliotecas utilizadas:**
~~~
# Instalação dos Pacotes.
!pip install colorama > /dev/null
!pip install catboost > /dev/null
!pip install category_encoders > /dev/null

# Importando as bibliotecas Pandas e Numpy.
import pandas as pd
import numpy as np

# Importando a função train_test_split para a divisão do nosso dataset em treino e teste.
from sklearn.model_selection import train_test_split

# Importando a classe Pipeline para simplificar o processo de pré-processamento.
from sklearn.pipeline import Pipeline

# Importando as bibliotecas necessárias para o tratamento dos dados.
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Importando os algoritmos de Machine Learning, da biblioteca Scikit-Learn, que iremos utilizar nesse projeto.
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Importando as bibliotecas para avaliação dos modelos.
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, auc

# Importando a pickle para serialização.
import pickle

# Importando as constantes definidas na biblioteca colorama que permitem alterar as cores e estilos de texto impresso no terminal.
from colorama import Fore, Style
~~~

### **6.3 Carregamento dos Dados:**
Primeiramente carregamos a base principal **application_train.csv** e fizemos a validação cruzada tipo Holdout, utilizando modo Out-of-Sample, dividindo a base entre 80% para treino e 20% para teste, resultando em: 

![**dataframe_shape_01**](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_01.png)
<br><br>
### **6.4 Tratamento e Transformação dos Dados**
Para a etapa de preparação dos dados utilizamos a classe **Pipeline** da biblioteca **Sklearn** para simplificar os processos e treinar os modelos de forma mais rápida, pois nesse momento o objetivo era visualizar apenas o resultado final, que é o desempenho dos modelos.

Identificamos as variáveis numéricas e as variáveis categóricas. Para as variáveis categóricas imputamos a **moda** **nos valores missings** e codificamos os dados com o **Target Encoder**. Para as variáveis numéricas imputamos a **média** **nos valores missings** e padronizamos os dados com **Standard Scaler**.
<br><br>
### **6.5 Treinamento dos Modelos**
O modelo baseline foi treinado com os algoritmos: DecisionTreeClassifier; LogisticRegression; RandomForestClassifier; GradientBoostingClassifier; LightGBM; XGBClassifier; XGBClassifier e CatBoostClassifier, da biblioteca Scikit-Learn, para no final avaliarmos as métricas de cada um e escolher a melhor opção dentro do que buscamos.
<br><br>
### **6.6 Resultado e Avaliação do Treinamento dos Modelos**

![métricas_modelo_baseline](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/metricas_modelo_baseline.png)

É importante observar todas as métricas e como elas se comportam para cada modelo treinado, porém nesse caso, como estamos tentando resolver um problema de concessão de crédito e risco de inadimplência, iremos focar na **AUC_ROC**, **Gini** e **KS**, porque além de serem métricas pedidas para serem priorizadas pelo time de negócios, são métricas que possuem uma boa capacidade discriminativa do modelo e sua capacidade de distinguir entre bons e maus pagadores, ajudando assim na tomada de decisão precisa e eficaz.

Então colocando um foco maior nessas métricas, podemos chegar à conclusão que o modelo treinado com o algoritmo **GradientBoostingClassifier** obteve melhor desempenho nos dados de teste, ante os demais modelos, além de ter métricas consistentes entre os conjuntos de treino e teste, que mostra que o modelo é robusto e confiável, capaz de generalizar bem para novos dados e fazer previsões precisas em situações do mundo real.

O modelo escolhido para ser o nosso modelo baseline (modelo de entrada), foi o modelo treinado com o algoritmo **GradientBoostingClassifier** e suas métricas servirão de parâmetro quando formos aplicar técnicas e métodos a fim de melhorar essas métricas.

![metricas_modelo_baseline_gb](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/metricas_modelo_baseline_gb.png)
<br><br>
### **6.7 Serialização do Modelo**
Para finalizar, salvei um arquivo pickle com o modelo baseline treinado, com o algoritmo GradientBoostingClassifier, na pasta de [artefatos](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/tree/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_baseline/artefatos), para o caso de ser necessário recuperá-lo ou colocá-lo em produção posteriormente.

<br><br>
## **7. Feature Engineering**
A Feature Engineering é um processo fundamental para ciência de dados. Ela se refere ao processo de criação e transformação de variáveis para melhorar o desempenho dos modelos de Machine Learning.

Pensando em obter um bom desempenho do modelo, optei por criar novas features com a intenção de agregar mais características ao treinamento do modelo.

Por se tratar de criação de variáveis, nessa etapa geralmente trabalhamos com muito volume de dados e grande demanda de processamento, por essa razão utilizei o **Spark** e **Spark.SQL** para essa etapa e separei o trabalho em sub etapas menores, conforme descrito abaixo.

Foram criadas ao todo 11.391 novas features, sendo que após o filtro e exclusão das variáveis com mais de 80% de valores nulos, restaram 6.017 features nos books criados.
<br><br>
### **7.1 Notebook:** [pos_cash_feature_engineering](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/pos_cash_feature_engineering.ipynb)
**Base utilizada:** POS_CASH_balance.csv

**Quantidade de variáveis criadas:** 800

**Nome do book:** book_pos_cash_balance
<br><br>
#### **7.1.1 Metodologia**
Já no ambiente PySpark comecei fazendo a leitura da base de dados **POS_CASH_balance.csv** e verificando número de linhas e colunas.

Foram criadas flags binárias com as janelas temporais de últimos 3 meses, últimos 6 meses, últimos 12 meses, últimos 24 meses e últimos 36 meses para poder agregar os dados por essas janelas e assim criar novas features com essas agregações.

A partir daí fiz a sumarização dos dados com agregações de soma, média, máximo e mínimo, para as janelas temporais criadas, pelo id **SK_ID_PREV**, resultando em 80 novas variáveis criadas.

Também criei variáveis com as mesmas agregações e janelas temporais, porém separando pelo status do contrato (NAME_CONTRACT_STATUS), resultando em mais 720 novas variáveis criadas.

Para finalizar fiz o Join das variáveis criadas e salvei o **book_pos_cash_balance** como um arquivo Parquet.
<br><br>
### **7.2 Notebook:** [installments_payments_feature_engineering](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/installments_payments_feature_engineering.ipynb)
**Base utilizada:** installments_payments.csv

**Quantidade de variáveis criadas:** 60

**Nome do book:** book_installments_payments
<br><br>
#### **7.2.1 Metodologia**

Já no ambiente PySpark comecei fazendo a leitura da base de dados **installments_payments.csv** e verificando número de linhas e colunas.

Foram criadas flags binárias com as janelas temporais de últimos 3 meses, últimos 6 meses, últimos 12 meses, últimos 24 meses e últimos 36 meses para poder agregar os dados por essas janelas e assim criar novas features com essas agregações.

A partir daí fiz a sumarização dos dados com agregações de soma, média, máximo e mínimo, para as janelas temporais criadas, pelo id **SK_ID_PREV**, resultando em 60 novas variáveis criadas.

Para finalizar salvei o **book_installments_payments** como um arquivo Parquet.
<br><br>
### **7.3 Notebook:** [credit_card_feature_engineering](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/credit_card_feature_engineering.ipynb)
**Base utilizada:** credit_card_balance.csv

**Quantidade de variáveis criadas:** 380

**Nome do book:** book_credit_card_balance
<br><br>
#### **7.3.1 Metodologia**

Já no ambiente PySpark comecei fazendo a leitura da base de dados **credit_card_balance.csv** e verificando número de linhas e colunas.

Foram criadas flags binárias com as janelas temporais de últimos 3 meses, últimos 6 meses, últimos 12 meses, últimos 24 meses e últimos 36 meses para poder agregar os dados por essas janelas e assim criar novas features com essas agregações.

A partir daí fiz a sumarização dos dados com agregações de soma, média, máximo e mínimo, para as janelas temporais criadas, pelo id **SK_ID_PREV**, resultando em 380 novas variáveis criadas.

Para finalizar salvei o **book_credit_card_balance** como um arquivo Parquet.
<br><br>
### **7.4 Notebook:** [previous_application_feature_engineering](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/previous_application_feature_engineering.ipynb)
**Base utilizada:** previous_application.csv

**Quantidade de variáveis criadas:** 7.700

**Nome do book:** book_previous_application
<br><br>
#### **7.4.1 Metodologia**

Já no ambiente PySpark comecei fazendo a leitura da base de dados **previous_application.csv** e verificando número de linhas e colunas.

Foram criadas flags binárias com as janelas temporais de últimos 3 meses, últimos 6 meses, últimos 12 meses, últimos 24 meses e últimos 36 meses para poder agregar os dados por essas janelas e assim criar novas features com essas agregações.

A partir daí fiz a sumarização dos dados com agregações de soma, média, máximo e mínimo, para as janelas temporais criadas, pelo id **SK_ID_CURR**, resultando em 300 novas variáveis criadas que foram salvas no **book_previous_application_01** como um arquivo Parquet.

Também criei variáveis com as mesmas agregações e janelas temporais, porém separando pelo status do contrato (NAME_CONTRACT_STATUS), resultando em mais 1.200 novas variáveis criadas que foram salvas no **book_previous_application_02** como um arquivo Parquet.

Depois carreguei os books **book_POS_CASH_balance**, **installments_payments** e **credit_card_balance**, fiz o Join desses books através da chave **SK_ID_PREV** e sumarizei fazendo as agregações pelas janelas temporais da **base previous_application.csv**, pelo id **SK_ID_CURR**, resultando em 6.200 novas variáveis criadas.

Por ser um processo que demanda muita capacidade de processamento e a minha capacidade disponível no momento ser limitada e não muito alta, precisei dividir esse último processo em 5 partes, cada parte para uma janela temporal, resultando então em 5 books de variáveis: **book_previous_application_03**, **book_previous_application_04**, **book_previous_application_05**, **book_previous_application_06**, **book_previous_application_07**.
<br><br>
### **7.5 Notebook:** [bureau_balance_feature_engineering](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/bureau_balance_feature_engineering.ipynb)
**Base utilizada:** bureau_balance.csv

**Quantidade de variáveis criadas:** 40

**Nome do book:** book_bureau_balance
<br><br>
#### **7.5.1 Metodologia**

Já no ambiente PySpark comecei fazendo a leitura da base de dados **bureau_balance.csv** e verificando número de linhas e colunas.

Foram criadas flags binárias com as janelas temporais de últimos 3 meses, últimos 6 meses, últimos 12 meses, últimos 24 meses e últimos 36 meses e também flags com o status da transação (STATUS) para poder agregar os dados por essas janelas e assim criar novas features com essas agregações.

A partir daí fiz a contagem de registros de cada status para as janelas temporais criadas, pelo id **SK_ID_BUREAU**, resultando em 40 novas variáveis criadas.

Para finalizar salvei o **book_bureau_balance** como um arquivo Parquet.
<br><br>
### **7.6 Notebook:** [bureau_feature_engineering](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/bureau_feature_engineering.ipynb)
**Base utilizada:** bureau.csv

**Quantidade de variáveis criadas:** 3.680

**Nome do book:** book_bureau
<br><br>
#### **7.6.1 Metodologia**

Já no ambiente PySpark comecei fazendo a leitura da base de dados **bureau.csv** e verificando número de linhas e colunas.

Foram criadas flags binárias com as janelas temporais de últimos 3 meses, últimos 6 meses, últimos 12 meses, últimos 24 meses e últimos 36 meses para poder agregar os dados por essas janelas e assim criar novas features com essas agregações.

A partir daí fiz a sumarização dos dados com agregações de soma, média, máximo e mínimo, para as janelas temporais criadas, pelo id **SK_ID_BUREAU**, resultando em 880 novas variáveis criadas.

Depois carreguei o **book_bureau_balance** e fiz o Join com as variáveis criadas a partir da base **bureau.csv**, através da chave **SK_ID_BUREAU**. Sumarizei todas as variáveis fazendo as agregações pelas janelas temporais da **bureau.csv**, pelo id **SK_ID_CURR**, resultando em 3.680 novas variáveis criadas.

Por ser um processo que demanda muita capacidade de processamento e a minha capacidade disponível no momento ser limitada e não muito alta, precisei dividir esse último processo em 2 partes, resultando então em 2 books de variáveis: **book_bureau_01**, **book_bureau_02**.
<br><br>
### **7.7 Notebook:** [application_train_feature_engineering](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/application_train_feature_engineering.ipynb)
**Base utilizada:** application_train.csv

**Quantidade de variáveis criadas:** 11

**Nome do book:** book_application_train
<br><br>
#### **7.7.1 Metodologia**

Já no ambiente PySpark comecei fazendo a leitura da base de dados **application_train.csv** e verificando número de linhas e colunas.

Optei por criar algumas novas variáveis a partir de outras existentes, para tentar trazer mais informações ao treinamento do modelo, resultando em 11 novas variáveis criadas que foram salvas no **book_application_train** como um arquivo Parquet.
<br><br>
**Variáveis criadas:**

**`APP_INCOME_CREDIT_RATIO`**: Resultado da divisão entre AMT_INCOME_TOTAL e AMT_CREDIT.

**`APP_ANNUITY_INCOME_RATIO`**: Resultado da divisão entre AMT_ANNUITY e AMT_INCOME_TOTAL.

**`APP_AGE`**: Resultado da divisão entre DAYS_BIRTH e -365.

**`APP_CHILDREN_INCOME_RATIO`**: Resultado da divisão entre CNT_CHILDREN e AMT_INCOME_TOTAL.

**`EXT_SOURCE_MEAN`**: Resultado da média das variáveis EXT_SOURCE_1, EXT_SOURCE_2 e EXT_SOURCE_3.

**`APP_EMPLOYED_LENGTH`**: Resultado da divisão entre DAYS_EMPLOYED e -365.

**`APP_GOODS_CREDIT_RATIO`**: Resultado da divisão entre AMT_GOODS_PRICE e AMT_CREDIT.

**`APP_CREDIT_DOWN_PAYMENT`**: Resultado da subtração da variável AMT_GOODS_PRICE pela variável AMT_CREDIT.

**`APP_NEW_DAYS_EMPLOYED_PERC`**: Resultado da divisão entre DAYS_EMPLOYED e DAYS_BIRTH.

**`APP_NEW_INCOME_PER_PERSON`**: Resultado da divisão entre AMT_INCOME_TOTAL e CNT_FAM_MEMBERS.

**`APP_NEW_PAYMENT_RATE`**: Resultado da divisão entre AMT_ANNUITY e AMT_CREDIT.
<br><br>
### **7.8 Notebook:** [feature_engineering_reducao](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/feature_engineering/feature_engineering_reducao.ipynb)
Trabalhar com um grande volume de variáveis pode demandar um alto poder de processamento e disponibilidade de tempo maior. Sabendo que geralmente variáveis com alta quantidade de valores nulos não são úteis para o treinamento de um modelo de machine learning, sendo assim optei por fazer um corte das variáveis com alto percentual de valores nulos, otimizando assim o tempo de processamento e reduzindo a necessidade de um alto poder de processamento.

O critério adotado para a exclusão da variável foi **excluir as variáveis com mais de 80% de valores nulos**. Optei também por excluir todas as variáveis que tenham cardinalidade igual a 1, caso elas existam.

Para essa etapa eu utilizei o Python com a biblioteca Pandas e dividi o trabalho por book de variáveis criados, resultando em 8 books de variáveis limpos, que nomeei como: **book_01**, **book_02**, **book_05**, **book_06**, **book_07**, **book_08**, **book_09**, **book_10**.

**Para informação:** Os books de variáveis **book_previous_application_03** e **book_previous_application_04** resultaram em zero variáveis após o filtro feito, portanto não os incluí na listagem acima.

Com essa limpeza saímos de 11.391 variáveis criadas para 6.017, uma redução de 52,82% de variáveis que não seriam úteis para o treinamento do nosso modelo, assim, a partir daqui, podemos otimizar o tempo de processamento das nossas tarefas.

Para consultar todos os books criados, acessar o diretório: [book_vars](https://drive.google.com/drive/folders/1VcuiqS9t8fiuCcG5iCDXoGQf1mrazaMd?usp=sharing)

<br><br>
## **8. Preparação dos Dados (DataPrep)**

A etapa de preparação dos dados é crucial para o sucesso da modelagem de machine learning, pois os dados precisam estar limpos, organizados e em um formato adequado para serem utilizados pelos algoritmos de aprendizado de máquina. Isso inclui garantir a qualidade dos dados, ajustar o formato para atender aos requisitos dos algoritmos, garantir eficiência computacional, promover interpretabilidade e explicabilidade dos modelos e assegurar generalização e robustez.

Como a partir de agora se tratava de um caso mais complexo, onde iriamos treinar o modelo que tenha um bom desempenho (superior ao do modelo baseline), foi feito o processo manualmente, passo por passo, sem o uso de pipeline, para ter um controle maior de cada etapa, caso seja necessário fazer algum ajuste específico, utilizar diferentes técnicas de pré-processamento e para permitir ter mais flexibilidade também.

Nesta etapa também fiz a seleção das variáveis utilizando algumas das técnicas mais usadas, para que o nosso modelo fosse treinado somente com as variáveis mais relevantes.

**Nota**: Para otimizar o processamento dos dados e consumir a menor quantidade de memória possível, pois tenho recursos limitados, fiz a preparação dos dados em etapas. Primeiro realizei o DataPrep de cada book de variável juntamente com a base principal e depois fiz a seleção de variáveis. Para finalizar, juntei todas as variáveis vindas de cada etapa de DataPrep anterior, para realizar um DataPrep final onde resultou em uma ABT de treino e uma ABT de teste, com 20 variáveis ao todo (incluindo as variáveis de ID e Target), salvas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing), prontas para serem usadas no treinamento do modelo. Também salvei arquivos pickle com os objetos resultantes da limpeza e transformação dos dados, na pasta [artefatos](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/tree/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/artefatos), para posterior reprodução caso necessário.

Para essa etapa utilizei o Python e os seguintes pacotes e bibliotecas.
~~~
# Instalação dos Pacotes.
!pip install colorama > /dev/null

# Importando as bibliotecas Pandas e Numpy.
import pandas as pd
import numpy as np

# Importando as constantes definidas na biblioteca colorama que permitem alterar as cores e estilos de texto impresso no terminal.
from colorama import Fore, Style

# Importando a pickle para serialização.
import pickle

# Importando a função train_test_split para a divisão do nosso dataset em treino e teste.
from sklearn.model_selection import train_test_split

# Importando os algoritmos de Machine Learning, da biblioteca Scikit-Learn, que iremos utilizar nesse projeto.
from sklearn.ensemble import GradientBoostingClassifier

# Importando a biblioteca Matplotlib para plotarmos gráficos que ajudarão no entendimento dos dados.
import matplotlib.pyplot as plt

# Importando interface para o coletor de lixo (Garbage Collector) do Python.
import gc
~~~
<br><br>
### **8.1 Notebook:** [DataPrep_RL_01](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/DataPrep_RL/DataPrep_RL_01.ipynb)
**Base utilizada:** application_train.csv, book_01, book_02 e book_10

**Quantidade de variáveis selecionadas:** 6

**Nome da ABT resultante do processo:** abt_train_fs_01.csv e abt_test_fs_01.csv
<br><br>
#### **8.1.1 Leitura, Join e Split dos Dados**
Já no ambiente Python comecei fazendo a leitura das bases de dados **application_train.csv**, **book_01**, **book_02** e **book_10**, já verificando a quantidade de linhas e colunas. Depois fiz o Join das bases e em seguida fiz o split dos dados utilizando o modo Out-of-Sample dividindo eles em 80% para treino e 20% para teste, resultando nos seguintes conjuntos:

![dataframe_shape_02](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_02.png)
<br><br>
#### **8.1.2 Remoção das Variáveis Constantes**
Uma variável constante não possui variabilidade, e os modelos de machine learning precisam de variabilidade nos dados para identificar padrões e fazer previsões precisas, portanto é uma variável inútil para o treinamento do modelo. Sendo assim, optei por excluir possíveis variáveis constantes.

Neste caso não havia variável constante a ser excluída.

\========================================================================================
**Variáveis constantes que foram excluídas:** \[\]

**Tamanho do DataFrame:** (172205, 1083)
\========================================================================================
<br><br>
#### **8.1.3 Tratamento dos Valores Nulos**
Para esse tratamento fiz a remoção das variáveis com mais de 80% de valores nulos, pois essas variáveis não costumam ser significativas para o treinamento do modelo e a imputação de valor para as variáveis com menos de 80% de valores nulos, usando a média para as variáveis numéricas e a palavra “VERIFICAR” para as variáveis categóricas.

\========================================================================================
**Variáveis que foram excluídas por alto percentual de nulos:** \['VL_MAX_AMT_ANNUITY_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_MED_CNT_PAYMENT_U6M_REFUSED_PREVIOUS_APP', 'VL_MED_CNT_PAYMENT_U6M_APPROVED_PREVIOUS_APP', 'VL_MIN_AMT_ANNUITY_U6M_REFUSED_PREVIOUS_APP', 'VL_MAX_AMT_ANNUITY_U6M_REFUSED_PREVIOUS_APP', 'VL_MED_AMT_ANNUITY_U6M_REFUSED_PREVIOUS_APP', 'VL_TOT_AMT_ANNUITY_U6M_REFUSED_PREVIOUS_APP', 'VL_MIN_AMT_ANNUITY_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_MAX_CNT_PAYMENT_U6M_APPROVED_PREVIOUS_APP', 'VL_MIN_CNT_PAYMENT_U6M_APPROVED_PREVIOUS_APP', 'VL_TOT_CNT_PAYMENT_U6M_REFUSED_PREVIOUS_APP', 'VL_MAX_CNT_PAYMENT_U6M_REFUSED_PREVIOUS_APP', 'VL_MIN_AMT_ANNUITY_U6M_APPROVED_PREVIOUS_APP', 'VL_MIN_CNT_PAYMENT_U6M_REFUSED_PREVIOUS_APP', 'VL_MED_AMT_ANNUITY_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_TOT_AMT_ANNUITY_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_TOT_AMT_ANNUITY_U6M_CANCELED_PREVIOUS_APP', 'VL_MED_AMT_ANNUITY_U6M_CANCELED_PREVIOUS_APP', 'VL_MAX_AMT_ANNUITY_U6M_CANCELED_PREVIOUS_APP', 'VL_MIN_AMT_ANNUITY_U6M_CANCELED_PREVIOUS_APP', 'VL_TOT_AMT_ANNUITY_U6M_APPROVED_PREVIOUS_APP', 'VL_MED_AMT_ANNUITY_U6M_APPROVED_PREVIOUS_APP', 'VL_TOT_CNT_PAYMENT_U6M_APPROVED_PREVIOUS_APP', 'VL_MAX_AMT_ANNUITY_U6M_APPROVED_PREVIOUS_APP', 'VL_MED_CNT_PAYMENT_U6M_CANCELED_PREVIOUS_APP', 'VL_MIN_CNT_PAYMENT_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_TOT_CNT_PAYMENT_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_MED_CNT_PAYMENT_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_MAX_CNT_PAYMENT_U6M_UNUSED_OFFER_PREVIOUS_APP', 'VL_TOT_CNT_PAYMENT_U6M_CANCELED_PREVIOUS_APP', 'VL_MIN_AMT_ANNUITY_U6M_PREVIOUS_APP', 'VL_MAX_AMT_ANNUITY_U6M_PREVIOUS_APP', 'VL_TOT_AMT_ANNUITY_U6M_PREVIOUS_APP', 'VL_MED_AMT_ANNUITY_U6M_PREVIOUS_APP', 'VL_MAX_CNT_PAYMENT_U6M_CANCELED_PREVIOUS_APP', 'VL_TOT_CNT_PAYMENT_U6M_PREVIOUS_APP', 'VL_MED_CNT_PAYMENT_U6M_PREVIOUS_APP', 'VL_MAX_CNT_PAYMENT_U6M_PREVIOUS_APP', 'VL_MIN_CNT_PAYMENT_U6M_PREVIOUS_APP', 'VL_MIN_CNT_PAYMENT_U6M_CANCELED_PREVIOUS_APP'\]

**Tamanho do DataFrame:** (172205, 1043)
\========================================================================================
<br><br>
#### **8.1.4 Separação das Variáveis Categóricas**
Para a Regressão Logística vamos precisamos categorizar as variáveis numéricas mais a frente, então nesse momento não há necessidade de codificar as variáveis categóricas e nem submeter elas à seleção de variáveis, pois não há um grande volume de variáveis categóricas em nosso conjunto de dados. Sendo assim, vou separei as variáveis para serem tratadas após a seleção das variáveis numéricas.

\========================================================================================
**Variáveis Categóricas:**

NAME_CONTRACT_TYPE
<br>
CODE_GENDER
<br>
FLAG_OWN_CAR
<br>
FLAG_OWN_REALTY
<br>
NAME_TYPE_SUITE
<br>
NAME_INCOME_TYPE
<br>
NAME_EDUCATION_TYPE
<br>
NAME_FAMILY_STATUS
<br>
NAME_HOUSING_TYPE
<br>
OCCUPATION_TYPE
<br>
WEEKDAY_APPR_PROCESS_START
<br>
ORGANIZATION_TYPE
<br>
FONDKAPREMONT_MODE
<br>
HOUSETYPE_MODE
<br>
WALLSMATERIAL_MODE
<br>
EMERGENCYSTATE_MODE
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Treino:** (172205, 1027)

**Tamanho do DataFrame com as variáveis categóricas – Base de Treino:** (172205, 17)
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Teste:** (43052, 1027)

**Tamanho do DataFrame com as variáveis categóricas - Base de Teste:** (43052, 17)
\========================================================================================

Salvei então as bases com as variáveis categóricas separadamente com os nomes: **abt_train_cat.csv** e **abt_test_cat.csv** na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing).
<br><br>
#### **8.1.5 Ajuste da Escala dos Dados**
Optei por não fazer o ajuste de escala dos dados neste momento, porque para o processo de seleção de variáveis utilizaremos algoritmo baseado em árvore que não necessita de padronização, e para a modelagem com a Regressão Logística iremos categorizar todas as variáveis numéricas.
<br><br>
#### **8.1.6 Seleção de Variáveis (Feature Selection)**
A etapa de seleção de variáveis contribui para a construção de modelos mais precisos, eficientes e interpretáveis, pois permite identificar e utilizar apenas as variáveis mais relevantes para prever o resultado desejado. Isso não apenas melhora a eficiência computacional e reduz o tempo de treinamento, mas também ajuda a evitar overfitting, aumenta a interpretabilidade dos modelos e facilita a identificação de padrões significativos nos dados.  

Existem diversos métodos para fazer essa seleção de variáveis. No caso dos problemas de classificação, podemos citar: Feature Importance, Boruta, RFE, IV - Information Value, entre outros.  

Neste projeto, optei por utilizar como método principal o **Feature Importance**, que é um método derivado de modelos específicos, e a formulação matemática para calcular essa importância pode variar dependendo do modelo. Ele nos dá uma ideia de quais variáveis têm maior impacto no modelo. Podemos usar alguns algoritmos para esse método, mas a escolha pode impactar tanto no desempenho, quanto no tempo de processamento dessa seleção. Para exemplificar, irei utilizar nesse momento o **GradientBoostingClassifier**, por ter uma boa performance. Posteriormente podemos fazer testes com o **RandomForestClassifier**, **DecisionTreeClassifier** e o **XGBoost**.  

**Outros métodos usados para seleção de variáveis são:** Eliminação de variáveis por % de valores nulos e por variância (por exemplo, variável constante). Nas etapas anteriores já foram feitos os tratamentos das variáveis com alto percentual de valores nulos e das variáveis constantes.

O valor de corte escolhido para selecionar as variáveis pelo Feature Importance foi de 0.05, ou seja, queremos selecionar as variáveis que têm uma importância maior que 5% da importância máxima encontrada.

O resultado da seleção de variáveis dessa etapa foi:

![vars_select_01](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_01.png)

Após selecionar as variáveis, salvei elas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_fs_01.csv** e **abt_test_fs_01.csv**.

<br><br>
### **8.2 Notebook:** [DataPrep_RL_02](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/DataPrep_RL/DataPrep_RL_02.ipynb)
**Base utilizada:** application_train.csv e book_05

**Quantidade de variáveis selecionadas:** 8

**Nome da ABT resultante do processo:** abt_train_fs_02.csv e abt_test_fs_02.csv
<br><br>
#### **8.2.1 Leitura, Join e Split dos Dados**
Já no ambiente Python comecei fazendo a leitura das bases de dados **application_train.csv** e **book_05**, já verificando a quantidade de linhas e colunas. Depois fiz o Join das bases e em seguida fiz o split dos dados utilizando o modo Out-of-Sample dividindo eles em 80% para treino e 20% para teste, resultando nos seguintes conjuntos:

![dataframe_shape_03](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_03.png)
<br><br>
#### **8.2.2 Remoção das Variáveis Constantes**
Realizado o mesmo processo da etapa anterior.

Neste caso não havia variável constante a ser excluída.

\========================================================================================
**Variáveis constantes que foram excluídas:** \[\]

**Tamanho do DataFrame:** (172205, 972)
\========================================================================================
<br><br>
#### **8.2.3 Tratamento dos Valores Nulos**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis que foram excluídas por alto percentual de nulos:** \[\]

**Tamanho do DataFrame:** (172205, 972)
\========================================================================================
<br><br>
#### **8.2.4 Separação das Variáveis Categóricas**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis Categóricas:**

NAME_CONTRACT_TYPE
<br>
CODE_GENDER
<br>
FLAG_OWN_CAR
<br>
FLAG_OWN_REALTY
<br>
NAME_TYPE_SUITE
<br>
NAME_INCOME_TYPE
<br>
NAME_EDUCATION_TYPE
<br>
NAME_FAMILY_STATUS
<br>
NAME_HOUSING_TYPE
<br>
OCCUPATION_TYPE
<br>
WEEKDAY_APPR_PROCESS_START
<br>
ORGANIZATION_TYPE
<br>
FONDKAPREMONT_MODE
<br>
HOUSETYPE_MODE
<br>
WALLSMATERIAL_MODE
<br>
EMERGENCYSTATE_MODE
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Treino:** (172205, 956)

**Tamanho do DataFrame com as variáveis categóricas – Base de Treino:** (172205, 17)
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Teste:** (43052, 956)

**Tamanho do DataFrame com as variáveis categóricas - Base de Teste:** (43052, 17)
\========================================================================================

Salvei então as bases com as variáveis categóricas separadamente com os nomes: **abt_train_cat.csv** e **abt_test_cat.csv** na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing).
<br><br>
#### **8.2.5 Ajuste da Escala dos Dados**
Realizado o mesmo processo da etapa anterior.
<br><br>
#### **8.2.6 Seleção de Variáveis (Feature Selection)**
Realizado o mesmo processo da etapa anterior.

O resultado da seleção de variáveis dessa etapa foi:

![vars_select_02](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_02.png)

Após selecionar as variáveis, salvei elas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_fs_02.csv** e **abt_test_fs_02.csv**.

<br><br>
### **8.3 Notebook:** [DataPrep_RL_03](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/DataPrep_RL/DataPrep_RL_03.ipynb)
**Base utilizada:** application_train.csv e book_06

**Quantidade de variáveis selecionadas:** 8

**Nome da ABT resultante do processo:** abt_train_fs_03.csv e abt_test_fs_03.csv
<br><br>
#### **8.3.1 Leitura, Join e Split dos Dados**
Já no ambiente Python comecei fazendo a leitura das bases de dados **application_train.csv** e **book_06**, já verificando a quantidade de linhas e colunas. Depois fiz o Join das bases e em seguida fiz o split dos dados utilizando o modo Out-of-Sample dividindo eles em 80% para treino e 20% para teste, resultando nos seguintes conjuntos:

![dataframe_shape_04](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_04.png)
<br><br>
#### **8.3.2 Remoção das Variáveis Constantes**
Realizado o mesmo processo da etapa anterior.

Neste caso não havia variável constante a ser excluída.

\========================================================================================
**Variáveis constantes que foram excluídas:** \[\]

**Tamanho do DataFrame:** (172205, 992)
\========================================================================================
<br><br>
#### **8.3.3 Tratamento dos Valores Nulos**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis que foram excluídas por alto percentual de nulos:** \[\]

**Tamanho do DataFrame:** (172205, 992)
\========================================================================================
<br><br>
#### **8.3.4 Separação das Variáveis Categóricas**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis Categóricas:**

NAME_CONTRACT_TYPE
<br>
CODE_GENDER
<br>
FLAG_OWN_CAR
<br>
FLAG_OWN_REALTY
<br>
NAME_TYPE_SUITE
<br>
NAME_INCOME_TYPE
<br>
NAME_EDUCATION_TYPE
<br>
NAME_FAMILY_STATUS
<br>
NAME_HOUSING_TYPE
<br>
OCCUPATION_TYPE
<br>
WEEKDAY_APPR_PROCESS_START
<br>
ORGANIZATION_TYPE
<br>
FONDKAPREMONT_MODE
<br>
HOUSETYPE_MODE
<br>
WALLSMATERIAL_MODE
<br>
EMERGENCYSTATE_MODE
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Treino:** (172205, 976)

**Tamanho do DataFrame com as variáveis categóricas – Base de Treino:** (172205, 17)
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Teste:** (43052, 976)

**Tamanho do DataFrame com as variáveis categóricas - Base de Teste:** (43052, 17)
\========================================================================================

Salvei então as bases com as variáveis categóricas separadamente com os nomes: **abt_train_cat.csv** e **abt_test_cat.csv** na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing).
<br><br>
#### **8.3.5 Ajuste da Escala dos Dados**
Realizado o mesmo processo da etapa anterior.
<br><br>
#### **8.3.6 Seleção de Variáveis (Feature Selection)**
Realizado o mesmo processo da etapa anterior.

O resultado da seleção de variáveis dessa etapa foi:

![vars_select_03](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_03.png)

Após selecionar as variáveis, salvei elas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_fs_03.csv** e **abt_test_fs_03.csv**.

<br><br>
### **8.4 Notebook:** [DataPrep_RL_04](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/DataPrep_RL/DataPrep_RL_04.ipynb)
**Base utilizada:** application_train.csv e book_07

**Quantidade de variáveis selecionadas:** 8

**Nome da ABT resultante do processo:** abt_train_fs_04.csv e abt_test_fs_04.csv
<br><br>
#### **8.4.1 Leitura, Join e Split dos Dados**
Já no ambiente Python comecei fazendo a leitura das bases de dados **application_train.csv** e **book_07**, já verificando a quantidade de linhas e colunas. Depois fiz o Join das bases e em seguida fiz o split dos dados utilizando o modo Out-of-Sample dividindo eles em 80% para treino e 20% para teste, resultando nos seguintes conjuntos:

![dataframe_shape_05](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_05.png)
<br><br>
#### **8.4.2 Remoção das Variáveis Constantes**
Realizado o mesmo processo da etapa anterior.

Neste caso não havia variável constante a ser excluída.

\========================================================================================
**Variáveis constantes que foram excluídas:** \[\]

**Tamanho do DataFrame:** (172205, 1002)
\========================================================================================
<br><br>
#### **8.4.3 Tratamento dos Valores Nulos**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis que foram excluídas por alto percentual de nulos:** \[\]

**Tamanho do DataFrame:** (172205, 1002)
\========================================================================================
<br><br>
#### **8.4.4 Separação das Variáveis Categóricas**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis Categóricas:**

NAME_CONTRACT_TYPE
<br>
CODE_GENDER
<br>
FLAG_OWN_CAR
<br>
FLAG_OWN_REALTY
<br>
NAME_TYPE_SUITE
<br>
NAME_INCOME_TYPE
<br>
NAME_EDUCATION_TYPE
<br>
NAME_FAMILY_STATUS
<br>
NAME_HOUSING_TYPE
<br>
OCCUPATION_TYPE
<br>
WEEKDAY_APPR_PROCESS_START
<br>
ORGANIZATION_TYPE
<br>
FONDKAPREMONT_MODE
<br>
HOUSETYPE_MODE
<br>
WALLSMATERIAL_MODE
<br>
EMERGENCYSTATE_MODE
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Treino:** (172205, 986)

**Tamanho do DataFrame com as variáveis categóricas – Base de Treino:** (172205, 17)
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Teste:** (43052, 986)

**Tamanho do DataFrame com as variáveis categóricas - Base de Teste:** (43052, 17)
\========================================================================================

Salvei então as bases com as variáveis categóricas separadamente com os nomes: **abt_train_cat.csv** e **abt_test_cat.csv** na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing).
<br><br>
#### **8.4.5 Ajuste da Escala dos Dados**
Realizado o mesmo processo da etapa anterior.
<br><br>
#### **8.4.6 Seleção de Variáveis (Feature Selection)**
Realizado o mesmo processo da etapa anterior.

O resultado da seleção de variáveis dessa etapa foi:

![vars_select_04](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_04.png)

Após selecionar as variáveis, salvei elas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_fs_04.csv** e **abt_test_fs_04.csv**.

<br><br>
### **8.5 Notebook:** [DataPrep_RL_05](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/DataPrep_RL/DataPrep_RL_05.ipynb)
**Base utilizada:** application_train.csv e book_08

**Quantidade de variáveis selecionadas:** 7

**Nome da ABT resultante do processo:** abt_train_fs_05.csv e abt_test_fs_05.csv
<br><br>
#### **8.5.1 Leitura, Join e Split dos Dados**
Já no ambiente Python comecei fazendo a leitura das bases de dados **application_train.csv** e **book_08**, já verificando a quantidade de linhas e colunas. Depois fiz o Join das bases e em seguida fiz o split dos dados utilizando o modo Out-of-Sample dividindo eles em 80% para treino e 20% para teste, resultando nos seguintes conjuntos:

![dataframe_shape_06](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_06.png)
<br><br>
#### **8.5.2 Remoção das Variáveis Constantes**
Realizado o mesmo processo da etapa anterior.

Neste caso não havia variável constante a ser excluída.

\========================================================================================
**Variáveis constantes que foram excluídas:** \[\]

**Tamanho do DataFrame:** (172205, 1484)
\========================================================================================
<br><br>
#### **8.5.3 Tratamento dos Valores Nulos**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis que foram excluídas por alto percentual de nulos:** \[\]

**Tamanho do DataFrame:** (172205, 1484)
\========================================================================================
<br><br>
#### **8.5.4 Separação das Variáveis Categóricas**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis Categóricas:**

NAME_CONTRACT_TYPE
<br>
CODE_GENDER
<br>
FLAG_OWN_CAR
<br>
FLAG_OWN_REALTY
<br>
NAME_TYPE_SUITE
<br>
NAME_INCOME_TYPE
<br>
NAME_EDUCATION_TYPE
<br>
NAME_FAMILY_STATUS
<br>
NAME_HOUSING_TYPE
<br>
OCCUPATION_TYPE
<br>
WEEKDAY_APPR_PROCESS_START
<br>
ORGANIZATION_TYPE
<br>
FONDKAPREMONT_MODE
<br>
HOUSETYPE_MODE
<br>
WALLSMATERIAL_MODE
<br>
EMERGENCYSTATE_MODE
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Treino:** (172205, 1468)

**Tamanho do DataFrame com as variáveis categóricas – Base de Treino:** (172205, 17)
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Teste:** (43052, 1468)

**Tamanho do DataFrame com as variáveis categóricas - Base de Teste:** (43052, 17)
\========================================================================================

Salvei então as bases com as variáveis categóricas separadamente com os nomes: **abt_train_cat.csv** e **abt_test_cat.csv** na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing).
<br><br>
#### **8.5.5 Ajuste da Escala dos Dados**
Realizado o mesmo processo da etapa anterior.
<br><br>
#### **8.5.6 Seleção de Variáveis (Feature Selection)**
Realizado o mesmo processo da etapa anterior.

O resultado da seleção de variáveis dessa etapa foi:

![vars_select_05](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_05.png)

Após selecionar as variáveis, salvei elas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_fs_05.csv** e **abt_test_fs_05.csv**.

<br><br>
### **8.6 Notebook:** [DataPrep_RL_06](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/DataPrep_RL/DataPrep_RL_06.ipynb)
**Base utilizada:** application_train.csv e book_09

**Quantidade de variáveis selecionadas:** 6

**Nome da ABT resultante do processo:** abt_train_fs_06.csv e abt_test_fs_06.csv
<br><br>
#### **8.6.1 Leitura, Join e Split dos Dados**
Já no ambiente Python comecei fazendo a leitura das bases de dados **application_train.csv** e **book_09**, já verificando a quantidade de linhas e colunas. Depois fiz o Join das bases e em seguida fiz o split dos dados utilizando o modo Out-of-Sample dividindo eles em 80% para treino e 20% para teste, resultando nos seguintes conjuntos:

![dataframe_shape_07](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_07.png)
<br><br>
#### **8.6.2 Remoção das Variáveis Constantes**
Realizado o mesmo processo da etapa anterior.

Neste caso não havia variável constante a ser excluída.

\========================================================================================
**Variáveis constantes que foram excluídas:** \[\]

**Tamanho do DataFrame:** (172205, 1516)
\========================================================================================
<br><br>
#### **8.6.3 Tratamento dos Valores Nulos**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis que foram excluídas por alto percentual de nulos:** \['VL_MIN_VL_TOT_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_TOT_VL_TOT_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_TOT_VL_TOT_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MED_VL_TOT_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MAX_VL_TOT_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MIN_VL_TOT_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_TOT_VL_MED_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MED_VL_MED_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MED_VL_MIN_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MIN_VL_MED_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_TOT_VL_MAX_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MED_VL_MAX_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MAX_VL_MAX_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MIN_VL_MAX_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MIN_VL_MIN_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_TOT_VL_TOT_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MED_VL_TOT_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MAX_VL_TOT_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MAX_VL_MIN_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_TOT_VL_MED_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MED_VL_MED_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MAX_VL_MED_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MIN_VL_MED_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_TOT_VL_MAX_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MED_VL_MAX_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MAX_VL_MAX_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_TOT_VL_MIN_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MED_VL_MIN_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MAX_VL_MIN_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MIN_VL_MIN_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MIN_VL_MAX_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MIN_VL_MIN_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MAX_VL_MED_AMT_ANNUITY_U24M_BAD_DEBT_BUREAU', 'VL_MED_VL_TOT_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MAX_VL_MIN_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MAX_VL_MIN_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MED_VL_MIN_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_TOT_VL_MIN_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MIN_VL_MAX_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MAX_VL_MAX_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MED_VL_MAX_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_TOT_VL_MAX_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MIN_VL_MED_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MAX_VL_MED_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MED_VL_MED_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_TOT_VL_MED_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MIN_VL_TOT_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MAX_VL_TOT_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MED_VL_TOT_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_TOT_VL_TOT_AMT_ANNUITY_U24M_ACTIVE_BUREAU', 'VL_MIN_VL_MIN_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_TOT_VL_MIN_AMT_ANNUITY_U24M_SOLD_BUREAU', 'VL_MAX_VL_MED_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MAX_VL_MAX_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MED_VL_MIN_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_TOT_VL_MIN_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MAX_VL_TOT_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MIN_VL_TOT_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_TOT_VL_MED_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MED_VL_MED_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MIN_VL_MAX_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MIN_VL_MED_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_TOT_VL_MAX_AMT_ANNUITY_U24M_CLOSED_BUREAU', 'VL_MED_VL_MAX_AMT_ANNUITY_U24M_CLOSED_BUREAU'\]

**Tamanho do DataFrame:** (172205, 1452)
\========================================================================================
<br><br>
#### **8.6.4 Separação das Variáveis Categóricas**
Realizado o mesmo processo da etapa anterior.

\========================================================================================
**Variáveis Categóricas:**

NAME_CONTRACT_TYPE
<br>
CODE_GENDER
<br>
FLAG_OWN_CAR
<br>
FLAG_OWN_REALTY
<br>
NAME_TYPE_SUITE
<br>
NAME_INCOME_TYPE
<br>
NAME_EDUCATION_TYPE
<br>
NAME_FAMILY_STATUS
<br>
NAME_HOUSING_TYPE
<br>
OCCUPATION_TYPE
<br>
WEEKDAY_APPR_PROCESS_START
<br>
ORGANIZATION_TYPE
<br>
FONDKAPREMONT_MODE
<br>
HOUSETYPE_MODE
<br>
WALLSMATERIAL_MODE
<br>
EMERGENCYSTATE_MODE
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Treino:** (172205, 1436)

**Tamanho do DataFrame com as variáveis categóricas – Base de Treino:** (172205, 17)
\========================================================================================


========================================================================================
**Tamanho do DataFrame com as variáveis numéricas – Base de Teste:** (43052, 1436)

**Tamanho do DataFrame com as variáveis categóricas - Base de Teste:** (43052, 17)
\========================================================================================

Salvei então as bases com as variáveis categóricas separadamente com os nomes: **abt_train_cat.csv** e **abt_test_cat.csv** na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing).
<br><br>
#### **8.6.5 Ajuste da Escala dos Dados**
Realizado o mesmo processo da etapa anterior.
<br><br>
#### **8.6.6 Seleção de Variáveis (Feature Selection)**
Realizado o mesmo processo da etapa anterior.

O resultado da seleção de variáveis dessa etapa foi:

![vars_select_06](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_06.png)

Após selecionar as variáveis, salvei elas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_fs_06.csv** e **abt_test_fs_06.csv**.

<br><br>
### **8.7 Notebook:** [DataPrep_RL_final](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/DataPrep_RL/DataPrep_RL_final.ipynb)
**Base utilizada:** abt_train_fs_01.csv, abt_train_fs_02.csv, abt_train_fs_03.csv, abt_train_fs_04.csv, abt_train_fs_05.csv, abt_train_fs_06.csv e abt_train_cat.csv

**Quantidade de variáveis selecionadas:** 18

**Nome da ABT resultante do processo:** abt_train_01.csv e abt_test_01.csv
<br><br>
#### **8.7.1 Pacotes e bibliotecas utilizadas**
~~~
# Instalação dos Pacotes.
!pip install colorama > /dev/null
!pip install category_encoders > /dev/null

# Importando as bibliotecas Pandas e Numpy.
import pandas as pd
import numpy as np

# Importando as constantes definidas na biblioteca colorama que permitem alterar as cores e estilos de texto impresso no terminal.
from colorama import Fore, Style

# Importando a pickle para serialização.
import pickle

# Importando os algoritmos de Machine Learning, da biblioteca Scikit-Learn, que iremos utilizar nesse projeto.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Importando a biblioteca Matplotlib e Seaborn para plotarmos gráficos que ajudarão no entendimento dos dados.
import matplotlib.pyplot as plt
import seaborn as sns

# Importanto as funções para codificação de variáveis.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
~~~

#### **8.7.2 Leitura e Join das tabelas**
Já no ambiente Python comecei fazendo a leitura das bases com as variáveis selecionadas nas etapas anteriores. Fiz o Join, somente das variáveis distintas, tanto para conjunto de treino quanto para conjunto de teste, resultando nos seguintes conjuntos:

\========================================================================================
**Tamanho do DataFrame de Treino:** (172205, 15)

**Tamanho do DataFrame de Teste:** (43052, 15)
\========================================================================================
<br><br>
#### **8.7.3 Seleção de Variáveis (Feature Selection)**
Usei novamente o método de **Feature Importance**, treinado com o **GradientBoostingClassifier**, para selecionar as melhores variáveis mais uma vez.  

O valor de corte escolhido para selecionar as variáveis pelo Feature Importance foi de 0.05, ou seja, queremos selecionar as variáveis que têm uma importância maior que 5% da importância máxima encontrada.

O resultado da seleção de variáveis dessa etapa foi:

![vars_select_07](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_07.png)

![grafico_feature_importance](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/grafico_feature_importance.png)

Para finalizar juntei a essas variáveis as variáveis categóricas que havia separado nas etapas anteriores.
<br><br>
#### **8.7.4 Mapa de Correlação de Pearson**
Ao plotar um mapa de correlação de Pearson, podemos visualizar de forma rápida e intuitiva as relações lineares entre as variáveis selecionadas. Uma alta correlação positiva entre duas variáveis sugere que elas variam juntas na mesma direção, enquanto uma alta correlação negativa indica que elas variam inversamente. Por outro lado, uma correlação próxima a zero indica que as variáveis têm pouca ou nenhuma relação linear entre si.

Essa análise é importante para identificar se existe alguma alta correlação entre as variáveis, o que pode indicar redundância ou multicolinearidade no conjunto de dados.

- **Redundância:** Variáveis altamente correlacionadas podem fornecer informações redundantes para o modelo, o que pode levar a uma sobrecarga de informações e aumentar o tempo de treinamento sem melhorar significativamente o desempenho do modelo.
- **Multicolinearidade:** A multicolinearidade ocorre quando duas ou mais variáveis estão altamente correlacionadas entre si, o que pode levar a problemas na interpretação dos coeficientes do modelo e na instabilidade das previsões.

![mapa_corr_pearson](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/mapa_corr_pearson.png)

Através do gráfico podemos notar que existem duas variáveis com alta correlação entre si, valor de 0.78, sendo assim excluí uma das variáveis utilizando como critério a variável com menor IV.

==========================================================================================
**Lista de variáveis para remover devido à alta correlação e baixo IV:** \['VL_MIN_AMT_PAYMENT_U24M_INSTALLMENTS_U36M_PREVIOUS_APP'\]
\==========================================================================================
<br><br>
#### **8.7.5 Verificação do Information Value (IV)**
O Information Value (IV) é uma medida estatística que avalia a força da relação entre uma variável independente e a variável dependente em um modelo de classificação. O IV é importante porque permite identificar quais variáveis têm maior poder preditivo e contribuição para o modelo, ajudando na priorização e seleção das melhores características para melhorar a precisão do modelo de machine learning. Quanto maior o IV de uma variável, maior é sua capacidade de discriminar entre as classes da variável dependente, o que indica sua importância na previsão do resultado desejado.

**Vale ressaltar:** Enquanto a Feature Importance é uma medida do quanto cada variável contribui para a capacidade de previsão do modelo, com base na relação de uma variável explicativa com as outras variáveis explicativas o IV é uma medida da força da relação entre uma variável explicativa e a variável target em um modelo. Ambas têm sua importância e usabilidade.

![information_value](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/information_value.png)

Podemos notar que 6 variáveis são inúteis para a predição, sendo assim excluí essas variáveis do conjunto.

==========================================================================================
**Lista de variáveis para remover devido ao baixo valor de IV:** \['HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE', 'FLAG_OWN_CAR', 'FONDKAPREMONT_MODE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_OWN_REALTY'\]
\==========================================================================================
<br><br>
#### **8.7.6 Exclusão de Variáveis Sensíveis**
Excluir variáveis sensíveis, como gênero, do modelo de crédito é uma prática ética importante para evitar discriminação injusta, respeitar a privacidade das pessoas e garantir conformidade legal. Isso ajuda o modelo a se concentrar em critérios financeiros relevantes, promovendo decisões de crédito mais justas e objetivas, construindo confiança e credibilidade nas instituições financeiras.

Excluí a variável “CODE_GENDER” por julgar ser uma variável sensível neste caso, resultando nos conjuntos finais:

![dataframe_shape_08](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_08.png)

Após selecionar as variáveis, salvei elas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_fs_final.csv** e **abt_test_fs_final.csv**.
<br><br>
#### **8.7.7 Tratamento das Variáveis Categóricas**
Para o tratamento das variáveis categóricas, optei por aplicar o Target Encoder nas variáveis com cardinalidade maior que 15.

==========================================================================================
**Lista de vars para Target Encoding:** \['ORGANIZATION_TYPE', 'OCCUPATION_TYPE'\]
\==========================================================================================

Já para as variáveis com cardinalidade menor ou igual a 15 eu verifiquei a ordenação da taxa de evento por categoria, certificando de agrupar categorias quando a ordenação não fosse a ideal, pois para a Regressão Logística é importante que as variáveis tenham uma boa ordenação, isso ajuda a garantir que o modelo capture as relações entre as variáveis independentes (preditoras) e a variável dependente (evento de interesse, como o não pagamento de um empréstimo) de forma eficaz.
<br><br>
#### **8.7.8 Tratamento das Variáveis Numéricas**
Ao treinar um modelo de crédito, como a Regressão Logística, é importante categorizar as variáveis numéricas primeiro, por diversos motivos. Categorizar variáveis numéricas transforma-as em variáveis categóricas, tornando mais fácil interpretar o efeito dessas variáveis no modelo. Isso proporciona uma interpretabilidade mais clara, pois pode-se entender o impacto das variáveis em termos de grupos ou categorias, em vez de valores contínuos.

Além disso, a categorização pode capturar padrões não lineares que podem ser perdidos ao tratar as variáveis como contínuas. Por exemplo, a relação entre a idade e a probabilidade de inadimplência pode não ser linear, mas ao categorizar a idade em faixas etárias, é possível capturar melhor essa relação.

Por fim, categorizar variáveis pode tornar o modelo mais estável e melhorar o desempenho computacional. Modelos lineares, como a regressão logística, se beneficiam da categorização para melhor desempenho.

Selecionei as variáveis numéricas com cardinalidade maior que 15 e categorizei elas em 5 faixas (bins), já verificando também a ordenação da taxa de evento por categoria, certificando de agrupar categorias quando a ordenação não fosse a ideal.
<br><br>
#### **8.7.9 Criação de Variáveis Dummies**
Após todas as variáveis se tornarem categóricas e de baixa cardinalidade, transformei as variáveis em variáveis dummies usando o OneHot Encoder.

Transformar variáveis categóricas em variáveis dummy é importante para o treinamento do modelo de crédito com Regressão Logística porque permite que o modelo seja compatível com a estrutura linear da regressão logística, simplifica a interpretação dos efeitos das categorias, preserva a informação original, evita suposições de ordinalidade entre as categorias e previne viés nas estimativas do modelo.
<br><br>
#### **8.7.10 Verificando a Matriz de Correlação de Pearson das Variáveis Dummies**
Novamente verifiquei a correlação das variáveis, agora das variáveis dummies, e encontrei 10 variáveis com alta correlação entre si (valor de mais de 0.70). Exclui então uma das variáveis de cada par.

\========================================================================================
**Variáveis excluídas por alta correlação:** \['TFB_NAME_INCOME_TYPE_Student', 'TFB_NAME_EDUCATION_TYPE_Higher education', 'TFB_NAME_TYPE_SUITE_Spouse, partner', 'TFT_EXT_SOURCE_1_3', 'TFT_EXT_SOURCE_3_3'\]

**Tamanho do DataFrame:** (172205, 55)
\========================================================================================

Após essa etapa de preparação dos dados, salvei as tabelas na pasta [abt](https://drive.google.com/drive/folders/1jeA3ewgRYIVOEphFevml1XgnnBTnbSGz?usp=sharing) como **abt_train_01.csv** e **abt_test_01.csv**.

<br><br>
## **9. Treinamento do Modelo (Regressão Logística)**
Nessa etapa eu treinei o modelo usando a técnica da Regressão Logística. Além de ser uma técnica que resulta em bons resultados, também é uma técnica facilmente explicável. O notebook com os códigos dessa etapa pode ser consultado em [Modelagem_RL_01](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_regressao_logistica/Modelagem_RL_01.ipynb).
<br><br>
### **9.1 Ambiente de Desenvolvimento**
Utilizei o ambiente Python do Google Colab para a manipulação dos dados e treinamento dos modelos.
<br><br>
### **9.2 Pacotes e bibliotecas utilizadas**
~~~
# Instalação dos Pacotes.
!pip install colorama > /dev/null

# Importando as bibliotecas Pandas e Numpy.
import pandas as pd
import numpy as np

# Importando a biblioteca Matplotlib para plotarmos gráficos que ajudarão no entendimento dos dados.
import matplotlib.pyplot as plt

# Importando as constantes definidas na biblioteca colorama que permitem alterar as cores e estilos de texto impresso no terminal.
from colorama import Fore, Style

# Importando a pickle para serialização.
import pickle

# Importando o pacote Statsmodels para o treinamento do nosso modelo com a Regressão Logística.
import statsmodels.api as sm

# Importando as bibliotecas para avaliação dos modelos.
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
~~~

### **9.3 Treinamento do Modelo**
Comecei lendo as ABTs **abt_train_01.csv** e **abt_test_01.csv,** depois treinei o meu modelo de Regressão Logística gerando o modelo treinado e o scorecard.
<br><br>
![scorecard](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/scorecard.png)

O scorecard fornece uma representação das variáveis, juntamente com seus coeficientes beta e p-valor, que ajudam a entender a importância relativa de cada variável na decisão de crédito. Isso permite uma interpretação direta e quantitativa do risco associado a cada característica, facilitando a tomada de decisões de crédito mais informadas e consistentes.

Ao final, salvei o modelo treinado como um arquivo pickle, para ser utilizado para colocar o modelo em produção.
<br><br>
### **9.4 Métricas do Modelo**
O resultado do treinamento foi bem satisfatório, com as métricas de AUC-ROC, Gini e KS superiores ao do modelo baseline e com uma boa ordenação das faixas de taxa de evento, concentrando as maiores taxas nos primeiros decis. Não foi notado overfitting ou underfitting.
<br><br>
#### **Principais métricas do modelo treinado:**

![metricas](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/metricas.png)

<br><br>
#### **Histograma do score:**

![hist_treino](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/hist_treino.png)

![hist_teste](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/hist_teste.png)

<br><br>
#### **Faixas da Taxa de Evento:**

![grafico_ordenacao](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/grafico_ordenacao.png)

![faixas_score_treino](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/faixas_score_treino.png)

![faixas_score_teste](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/faixas_score_teste.png)

<br><br>
#### **Análise das Faixas de Score e Apetite de Risco:**

<br><br>
**<p align="center">Apetite de risco de 8,0% (atual da carteira)</p>**

![analise_faixas_8](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_8.png)

<br><br>
**<p align="center">Apetite de risco de 5,0% (conservador)</p>**

![analise_faixas_5](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_5.png)

<br><br>
**<p align="center">Apetite de risco de 12,0% (ousado)</p>**

![analise_faixas_12](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_12.png)

<br><br>
## **10. Treinamento do Modelo (Desafiante)**
Na busca por melhorar a performance do modelo e apresentar uma segunda opção ao time de negócios, uma opção com métricas melhores, porém com menor explicabilidade, optei por treinar um novo modelo utilizando outros algoritmos de machine learning que não fossem a Regressão Logística. Para esse modelo “desafiante”, primeiramente, eu resolvi fazer a tunagem dos hiperparâmetros utilizando o Grid Search e comparar com os resultados obtidos utilizando o Optuna.
<br><br>
### **10.1 Ambiente de Desenvolvimento**
Utilizei o ambiente Python do Google Colab para a manipulação dos dados e treinamento dos modelos.
<br><br>
### **10.2 Preparação dos Dados**

**Pacotes e bibliotecas utilizadas:**
~~~
# Instalação dos Pacotes.
!pip install colorama > /dev/null
!pip install category_encoders > /dev/null

# Importando as bibliotecas Pandas e Numpy.
import pandas as pd
import numpy as np

# Importando as constantes definidas na biblioteca colorama que permitem alterar as cores e estilos de texto impresso no terminal.
from colorama import Fore, Style

# Importando a pickle para serialização.
import pickle

# Importando as bibliotecas necessárias para o tratamento dos dados.
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder

# Importando os algoritmos de Machine Learning, da biblioteca Scikit-Learn, que iremos utilizar nesse projeto.
from sklearn.ensemble import GradientBoostingClassifier

# Importando a biblioteca Matplotlib e Seaborn para plotarmos gráficos que ajudarão no entendimento dos dados.
import matplotlib.pyplot as plt
import seaborn as sns
~~~

Para a preparação dos dados eu fiz a leitura e o Join das tabelas **abt_train_fs_01.csv**, **abt_train_fs_02.csv**, **abt_train_fs_03.csv**, **abt_train_fs_04.csv**, **abt_train_fs_05.csv**, **abt_train_fs_06.csv** e **abt_train_cat.csv**, tanto para o treino quanto para o teste.

Como as variáveis já haviam passado pela limpeza e tratamento dos dados na etapa de preparação de dados para a Regressão Logística, não foi preciso fazer o processo novamente, foi preciso apenas fazer a transformação, pois agora, os algoritmos que foram usados necessitam dessas transformações. Apliquei então a padronização para as variáveis numéricas utilizando o **Standard Scaler**, e fiz a codificação das variáveis categóricas utilizando o **Target Encoder** para as variáveis com cardinalidade superior a 15 e **One-Hot Encoder** para as variáveis com cardinalidade menor ou igual a 15.

Passei as variáveis novamente pela seleção de variáveis, utilizando o método de **Feature Importance**, treinado com o algoritmo **GradientBoostingClassifier** e fazendo o corte por **0.05** da importância máxima.

O resultado da seleção das variáveis foi:

![vars_select_08](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/vars_select_08.png)

![grafico_feature_importance_02](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/grafico_feature_importance_02.png)
<br><br>
Verifiquei a Correlação de Pearson e o IV das variáveis, porém não foi necessário excluir nenhuma variável, excluí apenas a variável sensível “CODE_GENDER_M”, resultando nos seguintes conjuntos:

![dataframe_shape_08](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/dataframe_shape_08.png)

Após essa etapa de preparação dos dados, salvei as tabelas na pasta [abt](https://drive.google.com/drive/folders/1YdxLRTaYCoTcYQy4zD2pgxh-ilRG-AqN?usp=sharing) como **abt_train_01.csv** e **abt_test_01.csv** e os arquivos pickle na pasta [artefatos](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/tree/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_desafiante/artefatos).

O notebook com os códigos dessa etapa pode ser consultado em [DataPrep_D_Final](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_desafiante/DataPrep_D_final.ipynb).
<br><br>
### **10.3 Treinamento dos Modelos com Grid Search**

**Pacotes e bibliotecas utilizadas:**
~~~
# Instalação dos Pacotes.
!pip install colorama > /dev/null

# Importando as bibliotecas Pandas e Numpy.
import pandas as pd
import numpy as np

# Importando a biblioteca Matplotlib para plotarmos gráficos que ajudarão no entendimento dos dados.
import matplotlib.pyplot as plt

# Importando as constantes definidas na biblioteca colorama que permitem alterar as cores e estilos de texto impresso no terminal.
from colorama import Fore, Style

# Importando a pickle para serialização.
import pickle

# Importando os algoritmos de Machine Learning que iremos utilizar nesse projeto.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Importando a classe GridSearchCV para nos ajudar a encontrar os melhores hiperparâmetros para um modelo.
from sklearn.model_selection import GridSearchCV

# Importando a itertools para trabalhar com iteradores e operações de iteração de forma simplificada e eficiente.
import itertools

# Importando a time para calcular tempo de processamento.
import time

# Importando as bibliotecas para avaliação dos modelos.
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
~~~

Comecei lendo as ABTs **abt_train_01.csv** e **abt_test_01.csv,** depois treinei os modelos utilizando os seguintes algoritmos: **DecisionTreeClassifier**, **RandomForestClassifier**, **GradientBoostingClassifier**, **LightGBM** e **XGBClassifier**. Utilizei a técnica de tunagem de hiperparâmetros com o **Grid Search**, para selecionar os melhores hiperparâmetros para cada algoritmo. O critério de escolha dos melhores hiperparâmetros foi a taxa de **AUC-ROC**.

Após o treinamento dos modelos obtive os seguintes resultados:

![metricas_desafiante_grid](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/metricas_desafiante_grid.png)

Avaliando essas métricas podemos concluir que o modelo treinado com **XGBoost** foi o modelo com melhores métricas de **AUC-ROC**, **Gini** e **KS**, que foram as métricas pedidas para serem priorizadas pelo time de negócios.

Ao final, salvei o modelo treinado como um arquivo pickle na pasta de [artefatos](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/tree/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_desafiante/artefatos), para ser utilizado para colocar o modelo em produção.

<br><br>
\=======================================================================================
**RESULTADO DO TREINAMENTO DO MODELO**

**Total de Modelos Treinados:** 160
<br>
Fitting 5 folds for each of 32 candidates, totalling 160 fits

**Melhores Parâmetros:** {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 250, 'subsample': 0.9}

**Melhor AUC:** 0.76036526309076
\=======================================================================================

<br><br>
#### **Principais métricas do modelo treinado:**

![metricas_desafiante_grid_02](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/metricas_desafiante_grid_02.png)

<br><br>
![graficos_desafiante_grid](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/graficos_desafiante_grid.png)

<br><br>
#### **Faixas da Taxa de Evento:**

![grafico_ordenacao_desaf_grid](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/grafico_ordenacao_desaf_grid.png)

![faixas_score_desaf_grid](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/faixas_score_desaf_grid.png)

<br><br>
#### **Análise das Faixas de Score e Apetite de Risco:**

<br><br>
**<p align="center">Apetite de risco de 8,0% (atual da carteira)</p>**

![analise_faixas_8_desaf_grid](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_8_desaf_grid.png)

<br><br>
**<p align="center">Apetite de risco de 5,0% (conservador)</p>**

![analise_faixas_5_desaf_grid](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_5_desaf_grid.png)

<br><br>
**<p align="center">Apetite de risco de 12,0% (ousado)</p>**

![analise_faixas_12_desaf_grid](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_12_desaf_grid.png)
<br><br>
O notebook dessa etapa pode ser consultado em [Modelagem_D_GridSearch_01](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_desafiante/Modelagem_D_GridSearch_01.ipynb).
<br><br>
### **10.4 Treinamento dos Modelos com Optuna**

**Pacotes e bibliotecas utilizadas:**
~~~
# Instalação dos Pacotes.
!pip install colorama > /dev/null
!pip install optuna > /dev/null

# Importando as bibliotecas Pandas e Numpy.
import pandas as pd
import numpy as np

# Importando a biblioteca Matplotlib para plotarmos gráficos que ajudarão no entendimento dos dados.
import matplotlib.pyplot as plt

# Importando as constantes definidas na biblioteca colorama que permitem alterar as cores e estilos de texto impresso no terminal.
from colorama import Fore, Style

# Importando a pickle para serialização.
import pickle

# Importando os algoritmos de Machine Learning que iremos utilizar nesse projeto.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Importando o Optuna para nos ajudar a encontrar os melhores hiperparâmetros para um modelo.
import optuna

# Importando a classe StratifiedKFold do módulo de validação cruzada do scikit-learn.
from sklearn.model_selection import StratifiedKFold

# Importando o Standard Scaler para pré-processamento dos dados.
from sklearn.preprocessing import StandardScaler

# Importando a itertools para trabalhar com iteradores e operações de iteração de forma simplificada e eficiente.
import itertools

# Importando a time para calcular tempo de processamento.
import time

# Importando as bibliotecas para avaliação dos modelos.
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
~~~

Comecei lendo as ABTs **abt_train_01.csv** e **abt_test_01.csv,** depois treinei os modelos utilizando os seguintes algoritmos: **DecisionTreeClassifier**, **RandomForestClassifier**, **GradientBoostingClassifier**, **LightGBM** e **XGBClassifier**. Utilizei a técnica de tunagem de hiperparâmetros com o **Optuna**, para selecionar os melhores hiperparâmetros para cada algoritmo. O critério de escolha dos melhores hiperparâmetros foi a taxa de **AUC-ROC**.

Após o treinamento dos modelos obtive os seguintes resultados:

![metricas_desafiante_optuna](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/metricas_desafiante_optuna.png)

Avaliando essas métricas podemos concluir que o modelo treinado com **XGBoost** foi o modelo com melhores métricas de **AUC-ROC**, **Gini** e **KS**, que foram as métricas pedidas para serem priorizadas pelo time de negócios.

Ao final, salvei o modelo treinado como um arquivo pickle na pasta de [artefatos](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/tree/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_desafiante/artefatos), para ser utilizado para colocar o modelo em produção.

<br><br>
\=======================================================================================
**RESULTADO DO TREINAMENTO DO MODELO**

**Número de testes finalizados:** 22

**Melhor teste:**
<br>
- **Melhor AUC:** 0.7604734255972808

- **Melhores Parâmetros:**
  <br>
  lambda: 5.005789967925523
  <br>
  alpha: 5.672570918593209
  <br>
  max_depth: 5
  <br>
  subsample: 0.789747440596746
  <br>
  colsample_bytree: 0.5580235889283456
  <br>
  min_child_weight: 4.494619376548708
  <br>
  eta: 0.14894168483037754
  <br>
  gamma: 0.7544485084382566
  <br>
  grow_policy: depthwise
  <br>
  max_leaves: 821
  <br>
  max_bin: 433
  <br>
  scale_pos_weight: 3.8363533800992498

\=======================================================================================

<br><br>
#### **Principais métricas do modelo treinado:**

![metricas_desafiante_optuna_02](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/metricas_desafiante_optuna_02.png)

<br><br>
![graficos_desafiante_optuna](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/graficos_desafiante_optuna.png)

<br><br>
#### **Faixas da Taxa de Evento:**

![grafico_ordenacao_desaf_optuna](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/grafico_ordenacao_desaf_optuna.png)

![faixas_score_desaf_optuna](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/faixas_score_desaf_optuna.png)

<br><br>
#### **Análise das Faixas de Score e Apetite de Risco:**

<br><br>
**<p align="center">Apetite de risco de 8,0% (atual da carteira)</p>**

![analise_faixas_8_desaf_optuna](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_8_desaf_optuna.png)

<br><br>
**<p align="center">Apetite de risco de 5,0% (conservador)</p>**

![analise_faixas_5_desaf_optuna](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_5_desaf_optuna.png)

<br><br>
**<p align="center">Apetite de risco de 12,0% (ousado)</p>**

![analise_faixas_12_desaf_optuna](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/imagens/analise_faixas_12_desaf_optuna.png)
<br><br>
O notebook dessa etapa pode ser consultado em [Modelagem_D_Optuna_01](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Modelagem_de_Credito/pod-academy-analise-de-credito-para-fintech/modelo_desafiante/Modelagem_D_Optuna_01.ipynb).
<br><br>
### **10.5 Escolha do Melhor Modelo**
O resultado do treinamento foi bem satisfatório, com as métricas de AUC-ROC, Gini e KS superiores ao do modelo baseline e da Regressão Logística, com uma boa ordenação das faixas de taxa de evento, concentrando as maiores taxas nos primeiros decis. Não foi notado overfitting ou underfitting.

Quanto à escolha do método de tunagem de hiperparâmetros, as duas técnicas apresentaram métricas bem parecidas, porém a tunagem com o Optuna levou um ligeira vantagem nas métricas que estamos priorizando, além de ter um tempo de processamento menor, portanto, **escolhi o modelo utilizando o Optuna, para tunagem de hiperparâmetros, como o modelo desafiante**.

<br><br>
## **11. Conclusão**
Ao longo deste projeto, percorremos um processo abrangente de modelagem preditiva com o objetivo de desenvolver um modelo de crédito robusto e eficaz. Começamos com a criação de variáveis, depois com a preparação dos dados, passando pela seleção de variáveis e pela construção de modelos utilizando regressão logística e outros algoritmos de machine learning.

Na etapa de preparação dos dados, realizamos uma análise minuciosa das variáveis, tratamos os dados ausentes e asseguramos que estivessem prontos para o treinamento dos modelos. Utilizamos técnicas de padronização e codificação de variáveis categóricas, além de uma seleção cuidadosa de variáveis, com foco nas mais importantes para o modelo.

Posteriormente, treinamos dois conjuntos de modelos: um utilizando regressão logística e outro com algoritmos mais avançados, como Gradient Boosting, XGBoost e LightGBM. Para cada conjunto, utilizamos técnicas de tunagem de hiperparâmetros, tanto com Grid Search quanto com Optuna, buscando otimizar as métricas de desempenho, como AUC-ROC, Gini e KS.

Os resultados obtidos demonstraram melhorias significativas em relação ao modelo baseline, com métricas de desempenho superiores e uma boa ordenação das faixas de taxa de evento. Observamos também que o modelo treinado com XGBoost foi consistentemente superior, apresentando as melhores métricas em comparação aos outros algoritmos testados. Os modelos podem ser implementados em um ambiente de produção, auxiliando a instituição financeira na tomada de decisões de crédito mais informadas e consistentes.

<br><br>
## **12. Próximos Passos**
Apesar da capacidade de processamento limitada, foi possível treinar um bom modelo para esse projeto, atendendo às exigências do time de negócios, porém podemos seguir adiante com esse projeto, utilizando o ambiente em nuvem e o poder de processamento da AWS por exemplo, para criação de novas variáveis explicativas, utilização de outras técnicas de seleção de variáveis, experimentação de novos hiperparâmetros para melhorar ainda mais as métricas do modelo e colocá-lo em produção.

Podemos também acompanhar e monitorar o desempenho do modelo ao longo do tempo. Isso envolve avaliar regularmente as métricas de desempenho, como AUC-ROC, Gini e KS, para garantir que o modelo continue a fornecer previsões precisas. Além disso, é fundamental realizar uma validação out-of-time para verificar a robustez do modelo em relação aos dados futuros. Isso envolve reservar uma parte dos dados mais recentes, não utilizados no treinamento do modelo, e testar o desempenho do modelo nesses dados para garantir que ele seja generalizável e não esteja superajustado aos dados de treinamento.

A segmentação da carteira também é uma estratégia importante para compreender melhor o comportamento dos diferentes segmentos de clientes. Isso pode envolver a divisão da carteira em grupos com base em características demográficas, comportamentais ou de risco, e analisar o desempenho do modelo em cada segmento separadamente. Isso permite ajustar o modelo para melhor atender às necessidades e características específicas de cada grupo de clientes.
