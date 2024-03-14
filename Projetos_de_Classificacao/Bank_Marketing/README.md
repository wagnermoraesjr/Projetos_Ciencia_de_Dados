# **Projeto de Modelagem (Classificação) - Bank Marketing**
Neste projeto vamos resolver um problema de negócio de um banco português, que quer prever se o seu cliente se inscreverá (1/yes) ou não (0/no) para um depósito a prazo. Para isso faremos todas as etapas do processo de desenvolvimento de um modelo seguindo a metodologia CRISP-DM.
<br>
<br>

![Bank Marketing](fintech-investment-financial-internet-technology-concept.jpg)
<br>
<br>

## **Introdução**
Nos dias de hoje, o setor bancário enfrenta um cenário extremamente dinâmico e competitivo, no qual entender profundamente o comportamento do consumidor se tornou fundamental para o sucesso das instituições financeiras. Nesse contexto, o uso estratégico de dados e técnicas avançadas, como o machine learning, desempenha um papel crucial no desenvolvimento e na execução de campanhas de marketing eficazes.

A importância do uso de dados e machine learning para campanhas de marketing no setor bancário é incontestável. A vasta quantidade de informações disponíveis, que inclui transações financeiras, histórico de compras, comportamento online e offline, entre outros, oferece uma oportunidade única para os bancos entenderem as necessidades e preferências dos clientes em um nível individualizado e em tempo real.

Ao aplicar técnicas de machine learning a esses dados, os bancos podem extrair insights valiosos e prever comportamentos futuros dos clientes com uma precisão sem precedentes. Isso permite a personalização de ofertas e comunicações, garantindo que as campanhas de marketing sejam altamente relevantes e impactantes para cada segmento de clientes.
<br>
<br>

## **Entendimento do problema de Negócio**
Construir um **modelo que preveja se um cliente do banco se inscreverá (1/yes) ou não (0/no) para um depósito a prazo**, utilizando algumas técnicas de Machine Learning e avaliando qual se sai melhor, de acordo com algumas métricas estabelecidas.

Um depósito a prazo é uma forma de investimento em que o cliente deposita uma quantia de dinheiro por um período específico em troca de uma taxa de juros fixa ou variável, com o objetivo de garantir um retorno seguro e previsível sobre o capital investido. Já para o banco, os depósitos a prazo são uma parte importante da atividade bancária, oferecendo a eles uma fonte estável de financiamento de baixo custo, fortalecendo relacionamentos com os clientes e contribuindo para sua estabilidade financeira geral.
<br>
<br>

## **Entendimento dos Dados**
Os dados estão relacionados com campanhas de marketing direto de uma instituição bancária portuguesa. As campanhas de marketing foram baseadas em ligações telefônicas. Muitas vezes era necessário mais do que um contato com o mesmo cliente, para saber se o produto (depósito a prazo) seria ("sim") ou não ("não") inscrito.

Os datasets foram baixados do seguinte repositório: https://archive.ics.uci.edu/dataset/222/bank+marketing

Nesse projeto vamos trabalhar apenas com o dataset **bank-additional-full.csv** com todos os exemplos (41188 linhas e 21 colunas), ordenados por data (de maio de 2008 a novembro de 2010).
<br>
<br>

**Dicionário dos Dados:**
* **`age`**: Idade do cliente.
* **`job`**: Tipo de trabalho desempenhado pelo cliente.
* **`marital`**: O estado civil do cliente.
* **`education`**: Escolaridade do cliente.
* **`default`**: Tem crédito em inadimplência? (categórico: "não", "sim", "desconhecido").
* **`housing`**: Tem crédito à habitação? (categórico: "não", "sim", "desconhecido").
* **`loan`**: Tem empréstimo pessoal? (categórico: "não", "sim", "desconhecido").
* **`contact`**: Tipo de comunicação de contato.
* **`month`**: Último mês de contato do ano.
* **`day_of_week`**: Último dia de contato da semana.
* **`duration`**: Duração do último contato (em segundos). **Nota importante: este atributo afeta fortemente o alvo de saída (por exemplo, se duração = 0 então y = "no"). No entanto, a duração não é conhecida antes de uma chamada ser realizada. Além disso, após o término da chamada, y é obviamente conhecido. Assim, este input só deve ser incluído para fins de benchmark e deve ser descartado se a intenção for ter um modelo preditivo realista.**
* **`campaign`**: Quantidade de contatos realizados durante esta campanha e para este cliente.
* **`pdays`**: Número de dias que se passaram desde que o cliente foi contatado pela última vez em uma campanha anterior (999 significa que o cliente não foi contatado anteriormente).
* **`previous`**: Número de contatos realizados antes desta campanha e para este cliente.
* **`poutcome`**: Resultado da campanha de marketing anterior.
* **`emp.var.rate`**: Taxa de variação do emprego (indicador trimestral).
* **`cons.price.idx`**: Índice de preços ao consumidor (indicador mensal).
* **`cons.conf.idx`**: Índice de confiança do consumidor (indicador mensal).
* **`euribor3m`**: Taxa euribor a 3 meses (indicador diário).
* **`nr.employed`**: Número de empregados (indicador trimestral).
* **`y`**: O cliente efetuou um depósito a prazo? (binário: "sim" ou "não"). É a nossa variável target.
<br>

🚀 [**Notebook do Projeto**](https://github.com/wagnermoraesjr/Projetos_Ciencia_de_Dados/blob/main/Projetos_de_Classificacao/Bank_Marketing/Bank_Marketing.ipynb)
