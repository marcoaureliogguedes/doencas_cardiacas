DIAGNÓSTICO DE PROBLEMAS CARDÍACOS

Existem três tipos de interações com os médicos:
- Diagnóstico
- Tratamento
- Prognóstico

# Importando bibliotecas
import pandas as pd
import numpy as np 
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from protly.subplots import make_subplots

# Carregando os dados
df_datacardio = pd.read_csv("cardio_train.csv", sep=";")

# Visualizando as primeira linhas
df_datacardio.head()

# Visualiando as últimas linhas
df_datacardio.tail()

VISUALIZANÇÃO DE DADOS

Análise Exploratória

# Criando uma figura
fig = make_subplots(rows=4, cols=1)

# Verificando a distribuição das variáveis contínuas
fig.add_trace(go.Box(x=df_datacardio["age"] / 365, name="Idade"), row=1, col=1)
fig.add_trace(go.Box(x=df_datacardio["weight"], name="Peso"), row=2, col=1)
fig.add_trace(go.Box(x=df_datacardio["ap_hi"], name="Pressão sanguinea sistólica"), row=3, col=1)
fig.add_trace(go.Box(x=df_datacardio["ap_lo"], name="Pressão sanguinea diastólica"), row=4, col=1)

fig.update_layout(height=700)
fig.show()

# Usando a mesma estrutura para analisar as variáveis categóricas
fig = make_subplots(rows=2, cols=3)

fig.add_trace(go.Bar(y=df_datacardio["gender"].value_counts(), x=["Feminino", "Masculino"], name="Gênero"), row=1, col=1)
fig.add_trace(go.Bar(y=df_datacardio["cholesterol"].value_counts(), x=["Normal", "Acima do Normal", "Muito Acima do Normal"], name="Cholesterol"), row=1, col=2)
fig.add_trace(go.Bar(y=df_datacardio["gluc"].value_counts(), x=["Normal", "Acima do Normal", "Muito Acima do Normal"], name="Glicose"), row=1, col=3)
fig.add_trace(go.Bar(y=df_datacardio["smoke"].value_counts(), x=["Não Fumante", "Fumante"], name="Fumante"), row=2, col=1)
fig.add_trace(go.Bar(y=df_datacardio["alco"].value_counts(), x=["Não Alcoólatra", "Alcoólatra"], name="Alcoólatra"), row=2, col=2)
fig.add_trace(go.Bar(y=df_datacardio["active"].value_counts(), x=["Não Ativo", "Ativo"], name="Ativo"), row=2, col=3)

fig.update_layout(height=700)
fig.show()

# Qual é a distribuição de pessoas com doenças cardiacas
df_datacardio["cardio"].value_counts() / df_datacardio["cardio"].value_counts().sum()

# Clinado um mapa de correlação entre os dados
fig, ax = plt.subplots(figsize=(30, 10))
sns.heatmap(df_datacardio.corr(), annot=True, cmap="cyan")

Quais features têm maior correlação?

- gênero x altura
- peso x altura
- fumante x gênero
- índice de colesterol x glicose

CRIANDO O MODELO DE MACHINE LEARNING

# Retirando a variável ID
del df_datacardio["id"]

# Variável alvo
Y = df_datacardio["cardio"]
X = df_datacardio.loc[:, df_datacardio.columns != "cardio"]

# Imprimindo a variável
X 

# Utilizando o modelo Random Forest
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_teste = train_test_split(X, Y, test_size=0.33, random_state=42)

from sklearn.model_ensemble import RandomForestClassifier

ml_model = RandomForestClassifier()
ml_model.fit(x_train, y_train)

# Prevendo se o indivíduo tem a probabilidade de ter doença cardíaca
ml_model.predict(x_test.iloc[0].value_reshape(1, -1))

y_test.iloc[0]

from sklearn.metrics import	classification_report, confusion_matrix

predictions = ml_model.predict(x_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Avaliando o modelo
explainer = shap.TreeExplainer(ml_model)

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values[1], X)