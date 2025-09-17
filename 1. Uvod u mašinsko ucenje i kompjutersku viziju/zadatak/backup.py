# ovde importovati biblioteke
# staviti zelenu pozadinu i onda tako napraviti. 
import cv2
import numpy as np


def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca.

    Ova procedura se poziva automatski iz main procedure i taj deo koda nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca
    """
    # blood_cell_count = 0
    # TODO - Prebrojati crvena krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure
    
    # Učitavanje slike
    img = cv2.imread(image_path)
    if img is None:
        return 0

    # Gaussian blur - smanjenje šuma
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Pretvaranje slike u HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # HSV opseg za roze/crvenkaste eritrocite
    lower_pink = np.array([160, 30, 40])
    upper_pink = np.array([179, 220, 255])
    lower_pink2 = np.array([0, 30, 40])
    upper_pink2 = np.array([12, 220, 255])

    mask1 = cv2.inRange(hsv, lower_pink, upper_pink)
    mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morfološke operacije
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # -------------------------------
    # 1) Metod preko kontura
    # -------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        if 80 < area < 2000 and 0.6 < circularity < 1.3:
            contour_count += 1
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

    # -------------------------------
    # 2) Metod preko Hough krugova
    # -------------------------------
    circles = cv2.HoughCircles(mask,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=15,
                               param1=50,
                               param2=15,
                               minRadius=8,
                               maxRadius=30)

    hough_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            hough_count += 1
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)

    # -------------------------------
    # Kombinovani rezultat
    # -------------------------------
    # Ne sabiramo „na slepo“, da ne dupliramo previše
    # već uzimamo maksimum oba metoda
    blood_cell_count = max(contour_count, hough_count)

    # Debug prikaz
    cv2.putText(img, f"Broj zrnaca: {blood_cell_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Maska", mask)
    cv2.imshow("Detekcija", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return blood_cell_count

