1. Klasifikacija FashionMNIST skupa podataka:
    - Učitavanje podataka:

        ```python
        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        ```
    - Više o skupu podataka: https://keras.io/api/datasets/fashion_mnist/
2. Prepoznavanje emocije sa slike:
    1) Učitati CK+48 skup podataka
    2) Izračunati HOG deskriptore za slike 
    3) Klasifikovati emocije pomoću CNN
3. Klasifikacija otpada sa slike:
    1) Učitati TrashNet skup podataka
    2) Izračunati HOG deskriptore za slike 
    3) Klasifikovati emocije pomoću CNN
4. Detekcija konja na slikama: 
    Struktura podataka:
        - HorsesData/neg – 170 negativnih uzoraka (nema konja)
        - HorsesData/pos – 120 pozitivnih uzoraka (ima konja) + pripadajući groundtruth fajlovi sa bounding box koordinatama
        - HorsesData/test – 50 test slika + pripadajući groundtruth fajlovi
    
    Za svaku sliku je napravljena Groundtruth datoteka koja je imenovana po šablonu:
        - imeSlike__entires.groundtruth.
    U ovim datotekama se nalaze koordinate rezultujućeg bounding box-a na kome se nalazi konj. Koordinate su zadate u sledećem formatu:
        - top_left_x top_left_y bottom_right_x bottom_right_y

    Prilikom testiranja, za poređenje bounding box-ova koristiti Jaccard index.

    Pomoć:
        1) Obučiti CNN klasifikator koji razlikuje da li se na slici nalazi konj ili ne. 
        2) Napraviti algoritam za prolazak kroz sliku („sliding window“): isecati delove slike i slati ih CNN-u na klasifikaciju. Ako CNN prepozna konja – pamtimo koordinate.
        3) Spojiti sve isečke koje je CNN označio kao „konj“ u jedan region.
        4) Beležiti koordinate tog rezultujućeg bounding box-a.
        5) Evaluirati model koristeći Jaccard indeks za poređenje predikovanih i stvarnih bounding box-ova.



## Ekstrakcija HOG deskriptora (primer)

```python
import cv2


nbins = 9  # broj binova
cell_size = (8, 8)  # broj piksela po ćeliji
block_size = (3, 3)  # broj ćelija po bloku

# primer slike
img = cv2.imread('putanja/do/slike.jpg', cv2.IMREAD_GRAYSCALE)

hog = cv2.HOGDescriptor(
    _winSize=(img.shape[1] // cell_size[1] * cell_size[1],
              img.shape[0] // cell_size[0] * cell_size[0]),
    _blockSize=(block_size[1] * cell_size[1],
                block_size[0] * cell_size[0]),
    _blockStride=(cell_size[1], cell_size[0]),
    _cellSize=(cell_size[1], cell_size[0]),
    _nbins=nbins
)

img_descriptor = hog.compute(img)
```


