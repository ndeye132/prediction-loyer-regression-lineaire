import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
n = 120

surface = np.random.randint(20, 151, size=n)
pieces = np.clip((surface / 30 + np.random.randint(-1, 2, n)), 1, 6).astype(int)
quartier = np.random.choice([0, 1, 2], size=n, p=[0.45, 0.35, 0.20])
etage = np.random.randint(0, 6, size=n)

prix_m2 = np.where(quartier == 0, 900, np.where(quartier == 1, 1500, 2600))
loyer = 30000 + surface * prix_m2 + pieces * 8000 + etage * 3000 + np.random.normal(0, 15000, n)
loyer = (np.round(loyer / 1000) * 1000).clip(min=40000).astype(int)

df = pd.DataFrame({
    "Surface": surface,
    "Pieces": pieces,
    "Quartier": quartier,
    "Etage": etage,
    "Loyer": loyer,
})

df.to_csv("dataset_loyer.csv", index=False)

df = pd.read_csv("dataset_loyer.csv")

print("=== Aperçu des 5 premières lignes ===")
print(df.head())

print("\n=== Statistiques descriptives ===")
print(df.describe())

# Matrice de corrélation
plt.figure(figsize=(6, 5))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation entre les variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# Nuage de points Surface vs Loyer
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="Surface", y="Loyer", hue="Quartier", palette="viridis")
plt.title("Surface vs Loyer (par quartier)")
plt.xlabel("Surface (m²)")
plt.ylabel("Loyer (FCFA)")
plt.tight_layout()
plt.savefig("surface_vs_loyer.png", dpi=150)
plt.close()

print("\n=== Corrélations avec le Loyer ===")
print(correlation["Loyer"].sort_values(ascending=False))

#séparation des données en train et test
X = df[["Surface", "Pieces", "Quartier", "Etage"]]  # variables explicatives
y = df["Loyer"]                                       # variable à prédire
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Taille train : {X_train.shape[0]} logements")
print(f"Taille test : {X_test.shape[0]} logements")

# 2. Implémenter un modèle de régression linéaire multiple
model = LinearRegression()

# 3. Entraîner le modèle et afficher les coefficients
model.fit(X_train, y_train)

print("\n=== Coefficients de la régression ===")
for variable, coef in zip(X.columns, model.coef_):
    print(f"{variable} : {coef:.2f}")
print(f"constante du modéle: {model.intercept_:.2f}")


# 1. Faire des prédictions sur le groupe test
y_pred = model.predict(X_test)

# 2. Calculer les métriques
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Métriques d'évaluation ===")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# comparaison du  MAE à la moyenne du loyer du groupe test
print("\n===la moyenne du loyer du groupe test ===")
print(f"\nMoyenne du loyer (groupe test) : {y_test.mean():.2f}")

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Loyer réel (FCFA)")
plt.ylabel("Loyer prédit (FCFA)")
plt.title("Loyer réel vs Loyer prédit")
plt.tight_layout()
plt.savefig("loyer_reel_vs_predit.png", dpi=150)
plt.close()

print("\nGraphique enregistré : loyer_reel_vs_predit.png")
