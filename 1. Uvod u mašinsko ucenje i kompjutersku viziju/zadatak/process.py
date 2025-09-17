import cv2
import numpy as np

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca.
    Poboljšana verzija sa adaptivnim pragovima i boljom detekcijom boja.
    """

    # Učitavanje slike
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    # Kopija originalne slike za prikaz
    result_img = img.copy()
    
    # 1. PREDOBRADA SLIKE
    # Smanjenje šuma
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 2. KONVERZIJA U RAZLIČITE PROSTORE BOJA
    rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    
    # 3. DETEKCIJA ERITROCITA KROZ VIŠE PRISTUPA
    # Prvi pristup: RGB - crveni kanal
    r_channel = rgb[:,:,0]
    _, mask_r = cv2.threshold(r_channel, 100, 255, cv2.THRESH_BINARY)
    
    # Drugi pristup: HSV - crvena boja (oba opsega)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_hsv1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_hsv2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    
    # Treći pristup: LAB - a kanal (crveno-zelena osa)
    a_channel = lab[:,:,1]
    _, mask_a = cv2.threshold(a_channel, 130, 255, cv2.THRESH_BINARY)
    
    # 4. KOMBINOVANJE MASKİ
    combined_mask = cv2.bitwise_and(mask_r, mask_hsv)
    combined_mask = cv2.bitwise_and(combined_mask, mask_a)
    
    # 5. ZELENA POZADINA za vizuelizaciju
    green_bg = img.copy()
    green_bg[combined_mask == 0] = (0, 255, 0)
    
    # 6. MORFOLOŠKE OPERACIJE za čišćenje
    kernel = np.ones((3, 3), np.uint8)
    
    # Otvaranje za uklanjanje šuma
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Zatvaranje za popunjavanje rupa
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 7. PRONALAŽENJE KONTURA
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blood_cell_count = 0
    min_area = 30
    max_area = 1500
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Eritrociti su približno kružni
                if 0.4 < circularity < 1.6:
                    blood_cell_count += 1
                    
                    # Oznaka detektovanih eritrocita
                    cv2.drawContours(result_img, [cnt], -1, (0, 0, 255), 2)
                    
                    # Centar za broj
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(result_img, str(blood_cell_count), (cx-10, cy+5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 8. PRIKAZ REZULTATA
    cv2.putText(result_img, f"Ukupno: {blood_cell_count} eritrocita", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Prikaz svih koraka za debagovanje
    cv2.imshow("Originalna slika", img)
    cv2.imshow("Zelena pozadina", green_bg)
    cv2.imshow("RGB Mask", mask_r)
    cv2.imshow("HSV Mask", mask_hsv)
    cv2.imshow("LAB Mask", mask_a)
    cv2.imshow("Kombinovana maska", combined_mask)
    cv2.imshow("Ociscena maska", cleaned_mask)
    cv2.imshow("Rezultat detekcije", result_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return blood_cell_count