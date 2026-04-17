from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# LOAD MODEL
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
all_features = pickle.load(open('features.pkl', 'rb'))

model_features = [
    'TEMPERATURE', 'TURBIDITY', 'DISOLVED OXYGEN', 'pH',
    'AMMONIA', 'NITRATE', 'Population', 'Length', 'Weight'
]

@app.route('/')
def home():
    return render_template('index.html', features=all_features, show_result=False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data_all = []

        # Ambil semua input dari form
        for feature in all_features:
            value = request.form.get(feature)
            if value is None or value == "":
                return render_template(
                    'index.html',
                    prediction_text="❌ Semua input harus diisi",
                    features=all_features,
                    show_result=False
                )
            input_data_all.append(float(value))

        # Ambil data khusus untuk model
        input_data_model = [float(request.form.get(f)) for f in model_features]
        data = np.array([input_data_model])
        data_scaled = scaler.transform(data)

        # Prediksi model
        prediction = model.predict(data_scaled)

        # RULE ANOMALI
        anomali_rule = False
        alasan = []

        for i, feature in enumerate(model_features):
            val = input_data_model[i]
            fname = feature.lower()

            if 'ph' in fname:
                if val < 6.5:
                    anomali_rule = True
                    alasan.append("pH < 6.5")
                elif val > 8.5:
                    anomali_rule = True
                    alasan.append("pH > 8.5")

            if 'temp' in fname:
                if val > 35:
                    anomali_rule = True
                    alasan.append("Suhu > 35°C")

            if 'turbidity' in fname:
                if val > 50:
                    anomali_rule = True
                    alasan.append("Kekeruhan > 50 NTU")

        # CONFIDENCE
        try:
            proba = model.predict_proba(data_scaled)
            confidence = round(max(proba[0]) * 100, 2)
        except:
            confidence = None

        # KEPUTUSAN
        if anomali_rule:
            result = "ANOMALI"
            keterangan = "⚠️ Terjadi anomali karena: " + ", ".join(alasan)
        elif prediction[0] == 1 and confidence is not None and confidence > 85:
            result = "ANOMALI"
            keterangan = "⚠️ Anomali terdeteksi oleh model (confidence tinggi)"
        else:
            result = "NORMAL"
            keterangan = "✅ Kondisi air dalam batas normal"

        return render_template(
            'index.html',
            prediction_text=result,
            keterangan=keterangan,
            confidence=confidence,
            features=all_features,
            input_data=input_data_all,
            show_result=True
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"❌ Error: {str(e)}",
            features=all_features,
            show_result=False
        )

if __name__ == '__main__':
    app.run(debug=True)