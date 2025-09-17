Vaš zadatak je da analizirate podatke iz Titanic skupa podataka (titanic.csv) i kroz vizualizacije i tekst ispričate priču o putnicima broda.
Skup podataka sadrži informacije o putnicima, uključujući njihovu starost, pol, klasu putovanja, broj članova porodice na brodu, kartu, da li su preživeli ili ne.

Rečnik podataka:
---
Podatak  	Definicija	                                Vrednosti
survival	Preživljavanje	                            0 = Ne, 1 = Da
pclass	    Klasa karte 	                            1 = prva, 2 = druga, 3 = treća
sex	        Pol	
Age	        Starost u godinama	
sibsp	    Broj braće/sestara ili supružnika na brodu	
parch	    Broj roditelja/dece na brodu	
ticket	    Broj karte	
fare	    Cena karte	
cabin	    Broj kabine	
embarked	Luka ukrcavanja 	                         C = Cherbourg, Q = Queenstown, S = Southampton


Napomene o podacima:
---
pclass: Predstavlja približnu socio-ekonomsku klasu putnika (SES)
1. klasa = Visoka
2. klasa = Srednja
3. klasa = Niska

age: Starost je izražena decimalno ako je manje od 1 godine. Ako je starost procenjena, prikazana je u obliku xx.5

sibsp: Skup podataka definiše porodične veze na sledeći način:
- Sibling = brat, sestra, polubraća, polusestre
- Spouse = suprug, supruga (vanbračni partneri i verenici nisu uključeni)

parch: Skup podataka definiše porodične veze na sledeći način:
- Parent = majka, otac
- Child = ćerka, sin, pastorka, pastorko
- Neka deca su putovala samo sa dadiljom, pa im je parch=0