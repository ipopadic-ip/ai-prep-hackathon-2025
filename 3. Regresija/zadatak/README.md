Opis skupa podataka: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

Učitavanje skupa podataka:
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing(as_frame=True)
df = california.frame 

1. Izvršiti linearnu regresiju analitičkom metodom:   
    a) jednostruku linearnu regresiju (Isprobati za više atributa) 
    b) višestruku linearnu regresiju    
2. Izvršiti linearnu regresiju gradijentnom metodom:   
    a) jednostruku linearnu regresiju (Isprobati za više atributa)
    b) višestruku linearnu regresiju    
3. Proveriti L.I.N.E pretpostavke regresije za 2. zadatak.    
4. Implementirati linearnu regresiju (analitičku i gradijentnu metodu) bez upotrebe biblioteka