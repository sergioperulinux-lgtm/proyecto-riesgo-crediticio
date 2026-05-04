from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# =========================
# RUTA BASE (RAÍZ DEL PROYECTO)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# RUTAS A MODELOS (CARPETA /modelo)
# =========================
model_path = os.path.join(BASE_DIR, 'modelo', 'credit_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'modelo', 'scaler.pkl')
encoder_path = os.path.join(BASE_DIR, 'modelo', 'label_encoders.pkl')

# =========================
# CARGAR MODELOS
# =========================
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoder_path)

print("✅ Modelo, scaler y encoders cargados correctamente")

# =========================
# ENDPOINT DE PRUEBA
# =========================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API de Riesgo Crediticio funcionando 🚀"
    })

# =========================
# ENDPOINT PRINCIPAL
# =========================
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No se recibió JSON"}), 400

        df = pd.DataFrame([data])

        # =========================
        # VALIDACIÓN CAMPOS
        # =========================
        expected_columns = [
            'person_age',
            'person_income',
            'person_home_ownership',
            'person_emp_length',
            'loan_intent',
            'loan_grade',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_default_on_file',
            'cb_person_cred_hist_length'
        ]

        missing = [col for col in expected_columns if col not in df.columns]

        if missing:
            return jsonify({
                "error": f"Faltan campos: {missing}"
            }), 400

        df = df[expected_columns]

        # =========================
        # ENCODING (LABEL ENCODERS)
        # =========================
        for col in df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        # =========================
        # ESCALADO
        # =========================
        df_scaled = scaler.transform(df)

        # =========================
        # PREDICCIÓN
        # =========================
        prediction = model.predict(df_scaled)[0]

        try:
            probability = model.predict_proba(df_scaled)[0][1]
        except:
            probability = None

        risk = "HIGH RISK" if prediction == 1 else "LOW RISK"

        return jsonify({
            "risk": risk,
            "probability": float(probability) if probability is not None else None
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)