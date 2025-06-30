
# 🌞 Dynamic Solar Panel Tilt Angle Optimizer

This project predicts the **optimal tilt angle of a solar panel throughout the day** based on real-time weather conditions and sun elevation. The system uses a **machine learning model (Random Forest)** trained on synthetic weather-solar geometry data to suggest tilt angles that can **maximize solar energy absorption**.

---

## 🚀 Features

- 🌍 Real-time weather data fetched via the [Open-Meteo API](https://open-meteo.com/)
- 🧠 Trained `RandomForestRegressor` model for tilt prediction
- 🌤️ Inputs: temperature, humidity, wind speed, cloud cover, sun elevation, hour
- 📈 Simulation of optimal tilt angles from 5 AM to 7 PM (UTC)
- 📊 Dynamic visualization of tilt angles throughout the day using Matplotlib

---

## 📦 Dependencies

```bash
pip install requests numpy pandas scikit-learn matplotlib
```

---

## 📂 Project Structure

```
tilt_optimizer/
├── tilt_optimizer.py   # Main script
└── README.md           # Project overview
```

---

## 🔍 How It Works

1. **Fetch Real-time Weather**  
   Retrieves weather parameters like temperature, wind speed, cloud cover, and uses weather code to estimate humidity/cloudiness.

2. **Synthetic Training Data**  
   Generates 10,000 randomized weather+sun condition scenarios to train the model.

3. **Train Random Forest Model**  
   The model learns how tilt angles relate to environmental and time-based features.

4. **Simulate Day**  
   From 5 AM to 7 PM (UTC), simulates conditions at each hour, estimates solar elevation using solar geometry, and predicts optimal tilt.

5. **Plot**  
   Generates a line graph of tilt angles throughout the day.

---

## 🖼️ Example Output

_Visualization will appear here after execution._

---

## 📍 Run the Project

```bash
python tilt_optimizer.py
```

You can change the location by modifying the `latitude` and `longitude` values in the `main` section.

---

## 📌 To-Do

- [ ] Add support for historical weather data
- [ ] Integrate live solar irradiation calculations
- [ ] Build a simple web frontend using Streamlit or Flask

---
