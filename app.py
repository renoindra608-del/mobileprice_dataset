from flask import Flask, request, jsonify
import joblib
import pandas as pd

# 1. Inisialisasi aplikasi Flask (HARUS DI ATAS SEBELUM @app.route)
app = Flask(__name__)

# 2. Memuat model yang telah disimpan
joblib_model = joblib.load('gbr_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data JSON dari request
        json_data = request.get_json()
        
        # Mengambil isi dari list di dalam key 'data'
        data_df = pd.DataFrame(json_data['data'])
        
        # Melakukan prediksi menggunakan model GBR
        prediction = joblib_model.predict(data_df)
        
        # Mengembalikan hasil prediksi dalam bentuk JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)