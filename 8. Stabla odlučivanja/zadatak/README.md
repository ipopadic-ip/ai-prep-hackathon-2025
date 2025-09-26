1. Predikcija cene stanova:
    - Učitavanje skupa podataka:
    ```python
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
    ```
    - Uporediti sve algoritme (poželjno sa linearnom regresijom i neuronskim mrežama)

2. Klasifikacija pacijenata sa dijabetesom:
    - Učitavanje skupa podataka:
    ```python
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
    ```
    - Uporediti sve algoritme (poželjno sa linearnom regresijom i neuronskim mrežama)

3. Klasifikacija putnika na Titaniku u odnosu na preživljavanje
    - Učitati skup podataka
    - Primeni prethodnu obradu podataka (sa trećeg termina kursa)
    - Izvršiti klasifikaciju sa dosada rađenim algoritmima
    - Prikazati evaluaciju modela

4. Klasifikacija pacijenata koji imaju srčanih problema
    - skup podataka: heart.csv
    - Informacije o atributima:
        Age: starost pacijenta [godine]
        Sex: pol pacijenta [M: Muški, F: Ženski]
        ChestPainType: tip bola u grudima [TA: tipična angina, ATA: atipična angina, NAP: neanginalni bol, ASY: asimptomatski]
        RestingBP: krvni pritisak u mirovanju [mm Hg]
        Cholesterol: serumski holesterol [mg/dl]
        FastingBS: nivo glukoze u krvi nakon što osoba nije jela najmanje 8 sati [1: ako FastingBS > 120 mg/dl, 0: inače]
        RestingECG: rezultati EKG-a u mirovanju [Normal: normalan, ST: ST-T abnormalnosti (invertovane T talase i/ili ST elevacija ili depresija > 0.05 mV), LVH: verovatna ili definitivna hipertrofija leve komore po Estes kriterijumu]
        MaxHR: maksimalni dostignuti broj otkucaja srca [vrednost između 60 i 202]
        ExerciseAngina: angina izazvana vežbanjem [Y: da, N: ne]
        Oldpeak: ST depresija (oldpeak) [numerička vrednost]
        ST_Slope: nagib ST segmenta pri vrhu vežbanja [Up: uzlazni, Flat: ravan, Down: silazni]
        HeartDisease: ciljna klasa [1: prisustvo bolesti srca, 0: normalno]
        
5. Predikcija broja iznajmljenih bicikala na osnovu vremenskih i kalendarskih karakteristika
    - Više o skupu podataka: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
    - Učitavanje skupa podataka:
    ```python
        import pandas as pd
        from zipfile import ZipFile
        import requests
        from io import BytesIO

        # URL zip fajla
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

        # Preuzimanje fajla
        response = requests.get(url)
        zipfile = ZipFile(BytesIO(response.content))

        # Učitavanje 'hour.csv' iz zip fajla
        bike_hour = pd.read_csv(zipfile.open('hour.csv'))
        print(bike_hour.head())
    ```