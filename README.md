# Guia Prático de Classificação em Machine Learning
Autor: Danilo Martins Caldeira

## O que é Classificação?
A classificação é uma tarefa de aprendizado **supervisionado** onde o objetivo é treinar um modelo para prever categorias ou classes a partir de dados rotulados.
Por exemplo:

Prever se um e-mail é spam ou não spam.
Identificar se uma imagem contém um gato, cachorro ou pássaro.

### Etapas principais na classificação:
1. Coleta e preparação dos dados: Obter os dados e organizá-los (tratamento de valores ausentes, normalização, etc.).
2. Divisão do conjunto de dados: Separar os dados em treino e teste (exemplo: 80% treino, 20% teste).
3.  Treinamento do modelo: Usar algoritmos de classificação para ajustar o modelo aos dados de treino.
4. Avaliação: Validar o desempenho usando métricas como acurácia, F1-Score e matriz de confusão.
5. Predição: Usar o modelo treinado para prever classes em novos dados.

### Diferença entre Classificação e Regressão

| **Classificação**                           | **Regressão**                              |
|---------------------------------------------|--------------------------------------------|
| Prediz **categorias** ou **classes**.       | Prediz um **valor contínuo** ou **numérico**. |
| Exemplo: Diagnóstico de doenças (sim/não).  | Exemplo: Prever preços de casas.           |
| Métricas: Acurácia, matriz de confusão, etc.| Métricas: RMSE, MAE, R², etc.              |


### Principais Algoritmos de Classificação
1. Árvores de Decisão

Simples e interpretável.
Exemplo: DecisionTreeClassifier no Scikit-learn.

2. Random Forest

Conjunto de várias árvores de decisão para maior precisão.
Exemplo: RandomForestClassifier.

3. K-Nearest Neighbors (KNN)

Classifica com base nos vizinhos mais próximos.
Simples, mas pode ser lento em grandes conjuntos.

4. Support Vector Machines (SVM)

Cria hiperplanos para separar as classes.
Bom para dados de alta dimensionalidade.

5. Redes Neurais

Modelos mais complexos baseados na estrutura do cérebro humano.
Exemplo: MLPClassifier no Scikit-learn ou TensorFlow/Keras.

6. Naive Bayes

Baseado no teorema de Bayes, ótimo para texto e classificação probabilística.

7. Logistic Regression

Modelo estatístico para prever classes binárias.

8. XGBoost e LightGBM

Algoritmos avançados baseados em árvores de decisão. Muito eficientes em competições.


### Bibliotecas de Python para Classificação


1. Scikit-learn

Biblioteca mais popular para aprendizado supervisionado.
Algoritmos: SVM, Random Forest, KNN, etc.
Link: Scikit-learn

2. TensorFlow e Keras

Para criar redes neurais e modelos avançados.
Link: TensorFlow

3. PyTorch

Concorrente do TensorFlow, ótimo para aprendizado profundo.
Link: PyTorch

4. XGBoost e LightGBM

Frameworks poderosos para árvores de decisão e gradient boosting.
Link: XGBoost

5. Pandas e NumPy

Manipulação de dados para preparar os datasets.
Link: Pandas

6. Matplotlib e Seaborn

Visualização de dados.

7. Links: Matplotlib | Seaborn

### Bases de Dados Famosas para Classificação
- Iris Dataset

Classificação de espécies de flores.
Disponível no Scikit-learn.

- MNIST

Conjunto de dígitos manuscritos para reconhecimento de imagem.
Link: MNIST Dataset

- CIFAR-10

Imagens de 10 classes diferentes.
Link: CIFAR-10

- Titanic Dataset

Previsão de sobrevivência no naufrágio do Titanic.
Link: Titanic - Kaggle

- Breast Cancer Wisconsin

Dados médicos para prever diagnóstico de câncer.
Disponível no Scikit-learn.

- Wine Dataset

Classificação de tipos de vinho.
Disponível no Scikit-learn.

- Amazon Reviews

Análise de sentimentos em avaliações.
Link: Amazon Reviews Dataset

- Spam Base

Dados para classificar e-mails como spam ou não.
Link: Spam Base Dataset

## União entre Agrupamento e classificação. 
Podemos combinar a tarefa de agrupamento com a de classificação em cenários específicos onde queremos aproveitar a segmentação dos dados (agrupamento) como uma etapa prévia ou auxiliar para melhorar a classificação. Aqui estão algumas abordagens e exemplos práticos:


# O que é um Dataset?

Um **dataset** é um conjunto estruturado de dados, geralmente organizado em forma de tabela, onde cada linha representa uma **instância** ou **exemplo**, e cada coluna representa uma **feature** ou **variável**. Em problemas de machine learning, os datasets são usados para treinar, testar e validar modelos preditivos.

### Estrutura Típica de um Dataset

- **Linhas**: Cada linha do dataset representa uma **instância** ou **exemplo**. Em um problema de classificação, cada linha pode representar uma amostra de dados (como uma pessoa ou uma transação).
  
- **Colunas**: Cada coluna representa uma **feature** ou **variável** que descreve os dados. Por exemplo, em um dataset de dados de clientes, as colunas podem representar a **idade**, **renda**, **sexo**, entre outras variáveis.

- **Target**: Em problemas supervisionados, existe uma coluna adicional chamada **target** ou **rótulo**, que é a variável que estamos tentando prever ou classificar (por exemplo, "fraude" ou "não fraude", "sobreviveu" ou "não sobreviveu" no Titanic).

### Tipos de Datasets

- **Supervisionado**: O dataset contém exemplos de entrada (features) e saída (target), onde o objetivo é aprender a mapear as entradas para as saídas. Exemplo: Titanic (compreender a sobrevivência com base em atributos como idade, classe, etc.).

- **Não supervisionado**: O dataset contém apenas entradas (features), e o objetivo é descobrir padrões ou agrupamentos nos dados sem rótulos. Exemplo: segmentação de clientes em grupos.

- **Semi-supervisionado**: Parte dos dados tem rótulos e parte não tem. Esse tipo de dataset é comum quando é difícil ou caro rotular todos os dados manualmente.

- **Reforço**: Datasets utilizados em problemas de aprendizado por reforço, onde o modelo aprende através de interações com o ambiente e feedback de recompensas.

### Exemplo de Dataset

| Idade | Sexo | Renda  | Sobreviveu (Target) |
|-------|------|--------|---------------------|
| 22    | F    | 3000   | 0                   |
| 34    | M    | 4500   | 1                   |
| 25    | F    | 2200   | 0                   |
| 30    | M    | 3500   | 1                   |

Neste exemplo, as colunas **Idade**, **Sexo**, e **Renda** são as features, e **Sobreviveu** é o target (onde 0 indica "não sobreviveu" e 1 indica "sobreviveu").

### Importância dos Datasets
Os datasets são fundamentais para treinar modelos de machine learning. Eles fornecem a base sobre a qual os algoritmos aprendem padrões e fazem previsões. Além disso, a qualidade e diversidade dos dados no dataset têm um impacto direto no desempenho do modelo.



### Como unir agrupamento e classificação
1. Agrupamento como Pré-processamento para Classificação
O agrupamento pode ser usado para descobrir padrões ou segmentos em dados não rotulados antes de treinar um classificador.
Após identificar os grupos, podemos rotulá-los manualmente (se possível) ou usar os clusters como novas features para alimentar o modelo de classificação.
Exemplo:
Dataset: Dados de clientes sem rótulos.
Passos:
1. Aplique um algoritmo de agrupamento (como K-Means) para segmentar os clientes.
2. Rotule os clusters com base em características observadas (Ex.: "Clientes potenciais", "Clientes regulares").
3. Use esses rótulos como classes para treinar um modelo de classificação.

2. Classificação com Clusters como Features
Após realizar o agrupamento, as informações dos clusters (como a associação de cada ponto ao cluster ou a distância do ponto até o centro do cluster) podem ser adicionadas como features no modelo de classificação.
Exemplo:
Dataset: Dados de transações financeiras.
Passos:
1. Aplique agrupamento (como DBSCAN ou K-Means) para agrupar comportamentos similares.
2. Use a associação ao cluster como uma nova variável (Ex.: Cluster_ID).
3. Treine o modelo de classificação com as variáveis originais + Cluster_ID.

3. Identificação de Classes Ocultas para Expansão
Em algumas situações, as classes para classificação podem estar incompletas ou imprecisas.
O agrupamento ajuda a identificar subgrupos escondidos ou potenciais novas classes que podem ser adicionadas ao problema de classificação.
Exemplo:
Dataset: Diagnóstico médico (doença X ou não).
Passos:
1. Aplique agrupamento (Ex.: K-Means ou Hierárquico) para explorar subgrupos dentro de "doença X".
2. Identifique subclasses de "doença X" (Ex.: leve, moderada, grave).
3. Atualize o modelo de classificação para prever essas novas classes.

4. Classificação de Dados Não Rotulados Usando Agrupamento
- Quando não temos rótulos para os dados, o agrupamento pode atuar como uma solução temporária para rotular os dados automaticamente.
- Esses rótulos gerados podem ser usados como classes pseudo-rotuladas para treinar um modelo de classificação.
Exemplo:
Dataset: Dados de imagens não rotuladas.
Passos:
1. Use agrupamento (Ex.: K-Means) para segmentar as imagens em clusters.
2. Atribua os clusters como "classes" iniciais.
3. Treine um modelo de classificação com esses rótulos pseudo-rotulados.

---
# Principais Algoritmos de Classificação em Machine Learning

## 1. **Árvores de Decisão**

### Ideia Central  
As Árvores de Decisão utilizam uma estrutura em forma de árvore para dividir os dados em subconjuntos com base em condições, formando **ramos que levam a folhas representando classes finais**. Cada nó da árvore aplica uma regra de decisão baseada nos atributos.

### Pontos Positivos  
- Fácil de interpretar e visualizar.  
- Funciona bem com dados categóricos e numéricos.  
- Rápido para treinar em conjuntos menores.  

### Desvantagens  
- Pode superajustar os dados (overfitting) se não houver poda.  
- Sensível a pequenas variações nos dados.  

### Bibliotecas em Python  
- **Scikit-learn:** `DecisionTreeClassifier`  
- **Spark MLlib:** para grandes volumes de dados.  

### Formato da Base de Dados  
- A base de dados deve estar em um **formato tabular**, com as variáveis independentes (features) e a variável alvo (target) separadas.  

### Pseudo-código  
1. Definir os atributos do dataset (features) e a variável alvo (target).
2. Escolher o critério de divisão (ex.: Gini, Entropia).
3. Dividir o nó com base no critério escolhido:
   - Selecionar a feature com maior ganho de informação.
   - Criar ramos baseados nos valores dessa feature.
4. Repetir o processo para os nós filhos até:
   - Todos os nós serem puros (pertencerem a uma única classe) ou
   - A profundidade máxima ser atingida.
5. Utilizar a árvore resultante para classificação.

# 2. Random Forest

## Ideia Central  
O **Random Forest** é um algoritmo de aprendizado supervisionado que combina várias Árvores de Decisão, criando uma **floresta** de árvores. Cada árvore é gerada a partir de amostras diferentes do conjunto de dados (técnica de bootstrap) e considera apenas um subconjunto aleatório de features em cada divisão. A **classificação final é decidida pela votação da maioria** das árvores.

## Pontos Positivos  
- **Resistente ao overfitting** devido à combinação de múltiplas árvores.  
- **Desempenho robusto** em problemas complexos com muitos dados e variáveis.  
- Funciona bem mesmo com **dados ruidosos ou incompletos**.  
- **Suporte para dados categóricos e numéricos**.  

## Desvantagens  
- **Mais lento para treinar** em comparação com uma única árvore.  
- **Difícil de interpretar** devido à complexidade do conjunto de árvores.  
- **Consome mais memória**, especialmente para grandes florestas.  

## Bibliotecas em Python  
- **Scikit-learn**: `RandomForestClassifier`  
- **XGBoost**: implementações baseadas no Random Forest para grandes volumes de dados.  
- **LightGBM**: versão otimizada para grandes datasets.  

## Formato da Base de Dados  
A base de dados deve estar em **formato tabular**, contendo:  
- **Colunas das features**: representando as variáveis independentes (numéricas ou categóricas).  
- **Coluna do target**: variável dependente com os rótulos/classes a serem preditos.  

## Pseudo-código  
1. Dividir o dataset em subconjuntos de treino usando "bootstrap sampling".
2. Para cada subconjunto:
   - Criar uma Árvore de Decisão:
     a) Selecionar aleatoriamente um subconjunto de features.
     b) Dividir os nós com base no critério escolhido (Gini, Entropia, etc.).
3. Repetir o processo até gerar N árvores (tamanho da floresta).
4. Combinar as previsões de todas as árvores:
   - Para classificação, usar a votação da maioria.
5. Retornar a classe mais votada como resultado final.

# 3. K-Nearest Neighbors (KNN)

## Ideia Central  
O **K-Nearest Neighbors (KNN)** é um algoritmo baseado em instâncias que classifica um novo dado com base nos **K vizinhos mais próximos** no espaço das features. A proximidade é geralmente medida usando **distâncias métricas**, como a **distância Euclidiana**. A classe atribuída ao novo ponto é a **classe mais comum entre os K vizinhos**.

## Pontos Positivos  
- **Simples de implementar** e intuitivo.  
- **Flexível**: pode ser usado tanto para classificação quanto para regressão.  
- **Sem treinamento explícito**: o algoritmo simplesmente armazena os dados e realiza os cálculos no momento da classificação.  

## Desvantagens  
- **Computacionalmente caro**: para grandes datasets, pode ser lento, pois precisa calcular as distâncias para todos os pontos.  
- **Sensível a outliers**: vizinhos distantes podem influenciar o resultado.  
- **Depende da escala das variáveis**: é necessário normalizar ou padronizar os dados para evitar que variáveis com magnitudes maiores dominem o cálculo da distância.  

## Bibliotecas em Python  
- **Scikit-learn**: `KNeighborsClassifier`  
- **SciPy**: para cálculos de distância.  
- **MLlib**: implementações para grandes volumes de dados em Spark.  

## Formato da Base de Dados  
A base de dados deve estar em **formato tabular**, contendo:  
- **Colunas das features**: as variáveis independentes devem ser **normalizadas ou padronizadas**.  
- **Coluna do target**: variável dependente com os rótulos/classes a serem preditos.  

## Pseudo-código  
1. Definir o valor de K (número de vizinhos a considerar).
2. Para cada ponto a ser classificado:
   - Calcular a distância entre o ponto e todos os outros pontos do dataset.
   - Ordenar os vizinhos pela distância.
   - Selecionar os K vizinhos mais próximos.
3. Contar as classes dos K vizinhos mais próximos.
4. Atribuir ao ponto a classe mais frequente entre os vizinhos.

# 4. Logistic Regression

## Ideia Central  
A **Regressão Logística** é um modelo estatístico que utiliza uma função sigmoide (ou logística) para prever a **probabilidade de uma classe binária**. Ela transforma uma combinação linear das features em uma probabilidade, e um limiar (geralmente 0.5) é usado para classificar as instâncias.

### Fórmula da Função Sigmoide:  
\[
P(Y=1 | X) = \frac{1}{1 + e^{-(wX + b)}}
\]  
Onde:  
- \(w\) são os pesos das features.  
- \(b\) é o bias (intercepto).  

## Pontos Positivos  
- **Simples e eficiente** para problemas binários.  
- **Fácil de interpretar**: os coeficientes indicam a contribuição de cada feature na predição.  
- **Escalável** para grandes datasets.  
- Funciona bem para **dados lineares**.  

## Desvantagens  
- **Limitado a problemas lineares**: desempenho ruim em dados não linearmente separáveis.  
- **Sensível a outliers**: valores extremos podem influenciar os resultados.  
- Para classes desbalanceadas, os resultados podem ser enviesados.  

## Bibliotecas em Python  
- **Scikit-learn**: `LogisticRegression`  
- **Statsmodels**: para análises detalhadas de regressão logística.  
- **PyTorch** e **TensorFlow**: para implementações personalizadas.  

## Formato da Base de Dados  
A base de dados deve estar em **formato tabular**, contendo:  
- **Colunas das features**: variáveis independentes, de preferência **padronizadas ou normalizadas** para melhorar a convergência.  
- **Coluna do target**: variável dependente com rótulos binários (0 e 1).  

## Pseudo-código  
1. Inicializar os pesos e o bias do modelo.
2. Aplicar a função logística para calcular a probabilidade:
   P(Y=1 | X) = 1 / (1 + exp(-(wX + b)))
3. Definir o limiar (ex.: 0.5) para classificar as instâncias:
   - Se P >= 0.5, classe = 1.
   - Se P < 0.5, classe = 0.
4. Otimizar os pesos e o bias minimizando a função de perda:
   - Exemplo: Cross-Entropy Loss.
5. Iterar até que o modelo converja ou o número máximo de iterações seja atingido.

# 5. Support Vector Machines (SVM)

## Ideia Central  
O **Support Vector Machines (SVM)** é um algoritmo de classificação que busca encontrar um **hiperplano ótimo** que separa os dados em diferentes classes com a maior margem possível. Ele utiliza **vetores de suporte**, que são os pontos mais próximos do hiperplano, para determinar a melhor separação.

Para dados não linearmente separáveis, o SVM usa o **truque do kernel** para projetar os dados em um espaço de maior dimensão, tornando-os separáveis.

### Função do Hiperplano:  
\[
f(x) = w \cdot x + b
\]  
Onde:  
- \(w\) são os pesos das features.  
- \(b\) é o bias.  

## Pontos Positivos  
- **Eficaz para dados não linearmente separáveis** quando usado com kernels.  
- **Robusto contra overfitting**, especialmente em problemas de alta dimensionalidade.  
- Funciona bem em conjuntos de dados pequenos ou moderados.  

## Desvantagens  
- **Computacionalmente caro** para grandes conjuntos de dados.  
- **Sensível à escolha do kernel** e de parâmetros como \(C\) e \(\gamma\).  
- Difícil de interpretar em problemas com kernels complexos.  

## Bibliotecas em Python  
- **Scikit-learn**: `SVC` (para classificação)  
- **LIBSVM**: biblioteca subjacente usada por muitos pacotes.  
- **MLlib**: para grandes volumes de dados no Spark.  

## Formato da Base de Dados  
A base de dados deve estar em **formato tabular**, contendo:  
- **Colunas das features**: variáveis independentes numéricas. É recomendado **padronizar os dados**, pois o SVM é sensível à escala.  
- **Coluna do target**: variável dependente com rótulos das classes (binários ou multiclasses).  

## Pseudo-código  
1. Preparar os dados:
   - Normalizar ou padronizar as features.
   - Dividir o dataset em treino e teste.
2. Escolher o kernel apropriado (ex.: linear, RBF, polinomial).
3. Ajustar os parâmetros do modelo:
   - Parâmetro \(C\): controla o trade-off entre margem máxima e erro de classificação.
   - Parâmetro \(\gamma\): define a influência de um único ponto de treinamento (para kernels não lineares).
4. Treinar o modelo para encontrar o hiperplano ótimo.
5. Classificar novos pontos com base no lado do hiperplano em que eles se encontram.





