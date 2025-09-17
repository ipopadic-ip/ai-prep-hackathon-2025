import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje podataka
df = pd.read_csv("titanic.csv")

# Kratak pregled podataka
print("Prvih 5 redova:")
print(df.head())
print("\nInfo o kolonama:")
print(df.info())
print("\nStatistika:")
print(df.describe(include="all"))

# 1. Procenat preživelih
survival_rate = df['Survived'].value_counts(normalize=True) * 100
print("\nStopa preživljavanja (%):")
print(survival_rate)

# Vizuelizacija preživelih
sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Preživljavanje (0=Ne, 1=Da)")
plt.show()

# 2. Preživljavanje po polu
sns.countplot(x="Sex", hue="Survived", data=df, palette="Set1")
plt.title("Preživljavanje po polu")
plt.show()

# 3. Preživljavanje po klasi
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set3")
plt.title("Preživljavanje po klasi")
plt.show()

# 4. Distribucija godina
plt.figure(figsize=(10, 6))
sns.histplot(df["Age"].dropna(), bins=30, kde=True, color="blue")
plt.title("Distribucija starosti putnika")
plt.xlabel("Godine")
plt.ylabel("Broj putnika")
plt.show()

# 5. Kombinacija - godine i preživljavanje
plt.figure(figsize=(10, 6))
sns.boxplot(x="Survived", y="Age", data=df, palette="Set2")
plt.title("Starost u odnosu na preživljavanje")
plt.show()

# 6. Preživljavanje po luci ukrcavanja
sns.countplot(x="Embarked", hue="Survived", data=df, palette="coolwarm")
plt.title("Preživljavanje po luci ukrcavanja")
plt.show()

# 7. Priča u tekstu
print("\nPRIČA O TITANIKU:")
print("Na osnovu podataka vidimo da je stopa preživljavanja bila oko {:.2f}%.".format(survival_rate[1]))
print("Žene su imale mnogo veću šansu da prežive u odnosu na muškarce.")
print("Putnici prve klase su imali veću verovatnoću preživljavanja od putnika treće klase.")
print("Mlađi putnici i deca imali su nešto veću šansu da prežive u poređenju sa starijima.")
print("Takođe, polazak iz luke Cherbourg je povezan sa većom šansom za preživljavanje.")
