# Flights Price Prediction
Projekt predykcji cen biletów lotniczych przy użyciu modeli machine learning.

## Struktura katalogów
📂 flights_price_prediction  
 ┣ 📂 data                 
 ┃ ┣ 📄 processed_data.csv  
 ┣ 📂 source             L  
 ┃ ┣ 📄 random_forest.py  
 ┃ ┣ 📄 xgboost_model.py  
 ┃ ┣ 📄 compare.py         
 ┣ 📄 .gitignore  
 ┣ 📄 README.md  
 ┣ 📄 .requirements.txt

## Technologie
- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy

## Źródło
Inspiracja do projektu: https://www.kaggle.com/datasets/dilwong/flightprices

## Instalacja i uruchomienie

1. **Sklonuj repozytorium**  
 Aby sklonować repozytorium, użyj poniższego polecenia:
 ```sh
 git clone https://github.com/twoj-login/flights_price_prediction.git
 ```
2. **Przejdź do folderu projektu**
 ```sh
 cd flights_price_prediction
 ```

3. **Utwórz i aktywuj wirtualne środowisko**
 Jeśli chcesz używać wirtualnego środowiska, utwórz je za pomocą poniższego polecenia:

 Dla systemu Windows:
 ```sh
 venv\Scripts\activate
 ```
 Dla systemu Linux/macOS:
 ```sh
 source venv/bin/activate
 ```

4. **Zainstaluj wymagane biblioteki**
 ```sh
 pip install -r requirements.txt
 ```

5. **Ururchom modele**
 ```sh
 python source/random_forest.py
 ```
 ```sh
 python source/xgboost_model.py
 ```

6. **Uruchom porównanie wyników**
 ```sh
 python source/compare.py
 ```