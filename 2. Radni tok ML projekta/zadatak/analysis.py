import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje podataka
df = pd.read_csv("titanic.csv")

# 1. Provera duplikata
print("\nBroj duplikata:", df.duplicated().sum())
df = df.drop_duplicates()

# 2. Provera nedostajućih vrednosti
print("\nNedostajuće vrednosti po kolonama:")
print(df.isnull().sum())

# 3. Popunjavanje nedostajućih vrednosti
# Age – popunimo medianom (jer ima dosta outliera, pa median bolje "drži sredinu")
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked – popunimo najčešćom vrednošću
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fare – ako ima nedostajućih, popunimo medianom
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Cabin – ima previše praznih, obično se ili izbaci kolona ili se napravi oznaka "Unknown"
df["Cabin"] = df["Cabin"].fillna("Unknown")

# 4. Tipovi kolona
print("\nTipovi podataka:")
print(df.dtypes)

# -------------------------------
# Dodatne vizualizacije
# -------------------------------

# Scatter: Godine vs Cena karte, obojen po preživljavanju
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Age", y="Fare", hue="Survived", style="Sex", data=df, palette="Set1")
plt.title("Godine vs Cena karte (preživljavanje i pol)")
plt.show()

# Scatter: Godine vs Klasa
plt.figure(figsize=(10, 6))
sns.stripplot(x="Pclass", y="Age", hue="Survived", data=df, jitter=True, palette="Set2")
plt.title("Godine putnika po klasama i preživljavanju")
plt.show()

# Feature engineering: da vidimo ko su mame
# Mama = žena, Parch > 0 i Age > 18
df["IsMother"] = ((df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18))

print("\nBroj mama u podacima:", df["IsMother"].sum())

# Procena preživljavanja mama
mothers_survival = df[df["IsMother"]]["Survived"].mean() * 100
print(f"Procenat mama koje su preživele: {mothers_survival:.2f}%")

sns.countplot(x="IsMother", hue="Survived", data=df, palette="Set1")
plt.title("Preživljavanje - da li je žena bila mama")
plt.show()

# Scatter: Broj članova porodice (SibSp + Parch) vs Cena karte
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
plt.figure(figsize=(10, 6))
sns.scatterplot(x="FamilySize", y="Fare", hue="Survived", data=df, palette="coolwarm", alpha=0.7)
plt.title("Veličina porodice vs Cena karte (preživljavanje)")
plt.show()

# Boxplot: Cena karte po luci i preživljavanju
plt.figure(figsize=(10, 6))
sns.boxplot(x="Embarked", y="Fare", hue="Survived", data=df, palette="Set2")
plt.title("Cena karte po luci ukrcavanja i preživljavanju")
plt.show()

# Heatmap korelacija numeričkih kolona - 
# uvid kako su numeričke kolone povezane (npr. Fare i Pclass, Survived i Sex, itd.).
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelacija numeričkih vrednosti")
plt.show()

# Novi: FacetGrid Age vs Fare po polu
g = sns.FacetGrid(df, col="Sex", hue="Survived", height=5, palette="Set1")
g.map_dataframe(sns.scatterplot, x="Age", y="Fare", alpha=0.7)
g.add_legend()
plt.subplots_adjust(top=0.8)
g.figure.suptitle("Godine vs Cena karte po polu i preživljavanju")
plt.show()

# Novi: Violin plot Age vs Pclass
plt.figure(figsize=(10, 6))
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=df, split=True, palette="Set2")
plt.title("Starost po klasama i preživljavanju")
plt.show()

# Novi: Barplot FamilySize vs Survival rate
plt.figure(figsize=(10, 6))
family_survival = df.groupby("FamilySize")["Survived"].mean().reset_index()
sns.barplot(x="FamilySize", y="Survived", data=family_survival, palette="Set3")
plt.title("Preživljavanje po veličini porodice")
plt.ylabel("Stopa preživljavanja")
plt.show()

# -------------------------------
# Nastavlja se analiza kao i pre
# -------------------------------

# Kratak pregled podataka
print("\nPrvih 5 redova:")
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
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set1")
plt.title("Preživljavanje po klasi")
plt.show()

# 4. Distribucija godina
plt.figure(figsize=(10, 6))
sns.histplot(df["Age"], bins=30, kde=True, color="blue")
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

print("\nDODATNE PRIČE IZ VIZUALIZACIJA:")
print("- Cena karte i godine su imale značaj: skuplje karte i mlađi putnici češće su preživljavali, posebno žene.")
print("- Analiza mama pokazuje da su majke imale visoku stopu preživljavanja.")
print("- Veličina porodice je važna: male porodice (2-4 člana) imale su veću šansu, dok su sami putnici ili veoma velike porodice imali manju šansu.")
print("- FacetGrid je potvrdio razlike između muškaraca i žena: žene su preživele i kada su plaćale manje karte, dok su muškarci morali da pripadaju višoj klasi da bi imali šansu.")
print("- Violin plot pokazuje da su mlađi putnici u prvoj klasi preživljavali najčešće.")
print("- Korelacija potvrđuje da klasa i cena karte snažno utiču na ishod.")
