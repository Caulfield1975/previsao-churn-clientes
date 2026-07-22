import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Criando uma base de dados fictícia realista de Churn
np.random.seed(42)
n_samples = 1000

data = {
    "Tempo_Contrato_Meses": np.random.randint(1, 72, n_samples),
    "Valor_Mensal": np.random.uniform(30.0, 120.0, n_samples),
    "Suporte_Tecnico_Acionado": np.random.randint(0, 10, n_samples),
    "Atrasos_Pagamento": np.random.randint(0, 5, n_samples),
}

df = pd.DataFrame(data)

# Regra simulada para definir o Churn (clientes mais propensos a cancelar)
df["Churn"] = np.where(
    (df["Tempo_Contrato_Meses"] < 12)
    & (df["Atrasos_Pagamento"] > 2)
    | (df["Suporte_Tecnico_Acionado"] > 6),
    1,
    0,
)

# 2. Separação de Variáveis e Treino/Teste
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Treinamento do Modelo de Machine Learning
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Avaliação do Modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"--- Acurácia do Modelo: {acc * 100:.2f}% ---")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 5. Gerando o Gráfico da Matriz de Confusão
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Permanece", "Churn"],
    yticklabels=["Permanece", "Churn"],
)
plt.title("Matriz de Confusão - Previsão de Churn")
plt.xlabel("Previsto pelo Modelo")
plt.ylabel("Real")
plt.tight_layout()

# Salvando a imagem do gráfico
plt.savefig("confusion_matrix.png")
print("Gráfico 'confusion_matrix.png' gerado com sucesso!")