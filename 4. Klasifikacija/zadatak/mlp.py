# 1) Instalacija biblioteka (ako nije već)
#pip install -q scikit-image opencv-python scikit-learn seaborn tqdm

# 2) Importi
import os, glob, cv2, numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 3) Funkcija za učitavanje slika (pretpostavlja folder po klasama: /anger, /happy, /sad itd.)
def load_images_from_dir(root_dir, target_size=(48,48)):
    X, y = [], []
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_path in glob.glob(os.path.join(class_path, '*')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            if img.shape != target_size:
                img = cv2.resize(img, target_size)
            X.append(img)
            y.append(class_name)
    return np.array(X), np.array(y)

# 4) Učitaj dataset (promeni putanju na svoj dataset)
DATA_DIR = os.path.join(os.path.dirname(__file__), "CK+48")
X_imgs, y_labels = load_images_from_dir(DATA_DIR)
print("Broj slika:", X_imgs.shape[0])
print("Dimenzija jedne slike:", X_imgs[0].shape)

# 5) Label encoder (pretvori nazive emocija u brojeve)
le = LabelEncoder()
y_enc = le.fit_transform(y_labels)
print("Klase:", list(le.classes_))

# 6) HOG feature extraction
def compute_hog_features(X):
    feats = []
    for img in tqdm(X, desc="HOG extraction"):
        f = hog(img, orientations=9, pixels_per_cell=(8,8), 
                cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
        feats.append(f)
    return np.array(feats)

X_feats = compute_hog_features(X_imgs)
print("Dimenzija HOG vektora:", X_feats.shape)

# 7) Podela na train / val / test (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X_feats, y_enc, test_size=0.4, stratify=y_enc, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print("Train:", X_train.shape[0], "Val:", X_val.shape[0], "Test:", X_test.shape[0])

# 8) Standardizacija (fit na train)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

#----------------- Pocetak MLP ------------- #

from sklearn.neural_network import MLPClassifier

# 1) Definiši MLP (primer: 2 skrivena sloja sa po 256 neurona, ReLU aktivacija, Adam optimizer, broj epoha = 50)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=50,
    random_state=42,
    verbose=True
)

# 2) Treniraj na train
mlp_model.fit(X_train_s, y_train)

# 3) Evaluacija
val_acc_mlp = accuracy_score(y_val, mlp_model.predict(X_val_s))
test_acc_mlp = accuracy_score(y_test, mlp_model.predict(X_test_s))

print("\nRezultati MLP:")
print("Validation accuracy:", val_acc_mlp)
print("Test accuracy:", test_acc_mlp)

# 4) Izveštaj i konfuziona matrica
y_pred_mlp = mlp_model.predict(X_test_s)
print("\nMLP classification report:")
print(classification_report(y_test, y_pred_mlp, target_names=le.classes_))

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(7,6))
sns.heatmap(cm_mlp, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Purples")
plt.title("Confusion Matrix - MLP")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


#KRAJ MLP



# 9) SVM model (proba sa C=1, linear kernel)
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_s, y_train)
val_acc_svm = accuracy_score(y_val, svm_model.predict(X_val_s))
test_acc_svm = accuracy_score(y_test, svm_model.predict(X_test_s))

# 10) KNN model (n_neighbors=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_s, y_train)
val_acc_knn = accuracy_score(y_val, knn_model.predict(X_val_s))
test_acc_knn = accuracy_score(y_test, knn_model.predict(X_test_s))

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'pca__n_components': [50, 100, 200],
    'knn__n_neighbors': [1,3,5,7,9],
    'knn__weights': ['uniform','distance']
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)    # napomena: X_train su HOG feature-i, bez skaliranih transformacija (pipeline radi scaling)
print("\nBest params:", gs.best_params_)
best = gs.best_estimator_
print("Val accuracy (best):", best.score(X_val, y_val))
print("Test accuracy (best):", best.score(X_test, y_test))


# 11) Rezultati
print("\nRezultati SVM:")
print("Validation accuracy:", val_acc_svm)
print("Test accuracy:", test_acc_svm)

print("\nRezultati KNN:")
print("Validation accuracy:", val_acc_knn)
print("Test accuracy:", test_acc_knn)

# 12) Detaljniji izveštaj i konfuziona matrica (SVM primer)
y_pred_svm = svm_model.predict(X_test_s)
print("\nSVM classification report:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
