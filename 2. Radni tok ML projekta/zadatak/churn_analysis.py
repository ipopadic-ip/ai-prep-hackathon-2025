import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje podataka
states = pd.read_csv("../states_long_lat.csv")
telecom = pd.read_csv("../telecom_churn.csv")

print("States head:\n", states.head())
print("Telecom head:\n", telecom.head())

# Spajanje na osnovu kolone 'State'
merged = telecom.merge(states, on="State", how="left")

print("\nSpojeni dataset:")
print(merged.head())

# --- ANALIZE ---

# Churn znači odliv korisnika, tj. oni koji su odustali od usluge (prekinuli ugovor, prešli kod konkurencije).
# U tvom datasetu to je kolona Churn → True = korisnik je otišao, False = ostao.

# Dodatno - Prosečna širina i dužina za churn vs non-churn
geo_means = merged.groupby("Churn")[["Latitude", "Longitude"]].mean()
print("\nProsečne koordinate po churn statusu:")
print(geo_means)

# Dodatno - Vizuelizacija - scatter mapa
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged, x="Longitude", y="Latitude", hue="Churn", alpha=0.6, palette="coolwarm")
plt.title("Geografska raspodela churn vs non-churn korisnika")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# 1. Procenat churn-a
churn_rate = merged["Churn"].value_counts(normalize=True) * 100
print("\nStopa churn-a (%):")
print(churn_rate)

sns.countplot(x="Churn", data=merged, palette="coolwarm")
plt.title("Ukupan churn")
plt.show()

# 2. Churn po državama (top 10)
churn_by_state = merged.groupby("State")["Churn"].mean().sort_values(ascending=False) * 100
print("\nChurn po državama (%):")
print(churn_by_state.head(10))

plt.figure(figsize=(12, 6))
churn_by_state.head(10).plot(kind="bar", color="orange")
plt.title("Top 10 država po churn stopi")
plt.ylabel("Procenat churn-a")
plt.show()

# 3. Distribucija dnevnih minuta kod churn vs non-churn
plt.figure(figsize=(10, 6))
sns.kdeplot(data=merged, x="Total day minutes", hue="Churn", fill=True)
plt.title("Distribucija dnevnih minuta (churn vs non-churn)")
plt.show()

# 4. Prosečne vrednosti upotrebe po churn statusu
usage_cols = ["Total day minutes", "Total eve minutes", "Total night minutes", "Total intl minutes"]
mean_usage = merged.groupby("Churn")[usage_cols].mean()
print("\nProsečna upotreba po churn statusu:")
print(mean_usage)

mean_usage.T.plot(kind="bar", figsize=(10, 6))
plt.title("Prosečna upotreba usluga (churn vs non-churn)")
plt.ylabel("Prosek")
plt.show()

# 5. Churn u odnosu na Customer service calls
plt.figure(figsize=(10, 6))
sns.countplot(x="Customer service calls", hue="Churn", data=merged, palette="Set2")
plt.title("Churn u odnosu na pozive korisničkoj podršci")
plt.show()

# 6. Priča u tekstu
print("\nPRIČA O TELECOM CHURN-U:")
print("Ukupna stopa churn-a je oko {:.2f}%.".format(churn_rate[True] if True in churn_rate else 0))
print("Određene države imaju značajno viši churn od proseka, što može ukazivati na lokalne probleme ili konkurenciju.")
print("Korisnici koji su odustali obično koriste više minuta dnevno i u proseku češće zovu korisničku podršku.")
print("Vizuelizacije pokazuju da su distribucije minuta drugačije između churn i non-churn korisnika.")
