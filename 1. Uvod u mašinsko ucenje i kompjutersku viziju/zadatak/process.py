import cv2
import numpy as np

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca.
    Prilagodjena za svetlo i tamno roze/crevene eritrocite.
    """

    # Učitavanje slike
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    # Kopija originalne slike za prikaz
    result_img = img.copy()
    
    # 1. PREDOBRADA SLIKE - smanjenje šuma
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 2. KONVERZIJA U HSV (bolja za detekciju boja)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 3. DEFINISANJE OPSEGA BOJA ZA ERITROCITE
    # Svetlo roze (manje zasicenje)
    lower_pink = np.array([160, 30, 150])
    upper_pink = np.array([179, 100, 255])
    
    # Tamnije roze/crvena (visa saturacija)
    lower_red = np.array([0, 50, 150])
    upper_red = np.array([15, 255, 255])
    
    # 4. KREIRANJE MASKİ
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    # Kombinovanje maski
    combined_mask = cv2.bitwise_or(mask_pink, mask_red)
    
    # 5. ZELENA POZADINA za vizuelizaciju
    green_bg = img.copy()
    green_bg[combined_mask == 0] = (0, 255, 0)
    
    # 6. MORFOLOŠKE OPERACIJE za čišćenje maske
    kernel = np.ones((3, 3), np.uint8)
    
    # Otvaranje za uklanjanje šuma
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Zatvaranje za popunjavanje rupa u eritrocitima
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 7. PRONALAŽENJE KONTURA
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blood_cell_count = 0
    min_area = 50
    max_area = 1500
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Eritrociti su približno kružni
                if 0.5 < circularity < 1.5:
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
    cv2.imshow("Pink Mask", mask_pink)
    cv2.imshow("Red Mask", mask_red)
    cv2.imshow("Kombinovana maska", combined_mask)
    cv2.imshow("Ociscena maska", cleaned_mask)
    cv2.imshow("Rezultat detekcije", result_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return blood_cell_count