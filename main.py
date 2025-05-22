# --- Part 1: Libraries ---
import requests
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# --- Part 2: Fetch Weather Data ---
def fetch_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200 or 'current_weather' not in data:
        print("Error fetching weather data:", data)
        raise ValueError("Failed to fetch weather data from Open-Meteo.")

    current = data['current_weather']
    
    temp = current['temperature']
    wind_speed = current['windspeed']
    weather_code = current['weathercode']

    # Approximate clouds and humidity based on weather code
    # (you can fine-tune this mapping if you want)
    if weather_code in [0, 1]:  # clear sky, mainly clear
        clouds = 0
        humidity = 30
    elif weather_code in [2, 3]:  # partly cloudy, overcast
        clouds = 50
        humidity = 50
    elif weather_code in [45, 48]:  # fog
        clouds = 90
        humidity = 95
    elif weather_code in [51, 53, 55, 61, 63, 65]:  # rain
        clouds = 90
        humidity = 90
    else:
        clouds = 70
        humidity = 70

    return {
        "temperature": temp,
        "humidity": humidity,
        "clouds": clouds,
        "wind_speed": wind_speed,
    }

# --- Part 3: Generate Training Data ---
def generate_training_data():
    np.random.seed(42)
    X = []
    y = []
    for _ in range(10000):
        temp = np.random.uniform(10, 45)
        humidity = np.random.uniform(10, 100)
        clouds = np.random.uniform(0, 100)
        wind = np.random.uniform(0, 15)
        sun_elevation = np.random.uniform(0, 90)
        hour = np.random.randint(5, 19)

        if sun_elevation > 50:
            tilt = 10 + np.random.normal(0, 2)
        elif sun_elevation > 20:
            tilt = 30 + np.random.normal(0, 3)
        else:
            tilt = 50 + np.random.normal(0, 5)

        tilt = np.clip(tilt, 0, 90)

        X.append([temp, humidity, clouds, wind, sun_elevation, hour])
        y.append(tilt)

    return np.array(X), np.array(y)

# --- Part 4: Model Training ---
def train_model():
    X, y = generate_training_data()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- Part 5: Predict Tilt ---
def predict_tilt(model, features):
    X = np.array([
        features['temperature'],
        features['humidity'],
        features['clouds'],
        features['wind_speed'],
        features['sun_elevation'],
        features['hour']
    ]).reshape(1, -1)
    tilt = model.predict(X)[0]
    return tilt

# --- Part 6: Simulate a Whole Day ---
def simulate_day(model, lat, lon):
    times = []
    tilts = []
    now = datetime.utcnow().replace(hour=5, minute=0, second=0, microsecond=0)  # Start at 5 AM UTC

    for i in range(15):  # 5 AM to 7 PM (15 hours)
        temp_features = fetch_weather_data(lat, lon)
        temp_features['hour'] = (5 + i) % 24
        
        # Estimate elevation manually for each hour
        hour_angle = 15 * (temp_features['hour'] - 12)
        declination = 23.45 * math.sin(math.radians(360/365*(now.timetuple().tm_yday-81)))
        latitude_rad = math.radians(lat)
        declination_rad = math.radians(declination)
        elevation_angle = math.degrees(math.asin(math.sin(latitude_rad)*math.sin(declination_rad) +
                                                 math.cos(latitude_rad)*math.cos(declination_rad)*math.cos(math.radians(hour_angle))))
        temp_features['sun_elevation'] = max(elevation_angle, 0)
        
        tilt = predict_tilt(model, temp_features)
        times.append(temp_features['hour'])
        tilts.append(tilt)

    return times, tilts


# --- Main Execution ---
if __name__ == "__main__":
    try:
        latitude = float(input("Enter latitude: "))
        longitude = float(input("Enter longitude: "))

        print("Training model...")
        model = train_model()

        print("Simulating day...")
        hours, tilts = simulate_day(model, latitude, longitude)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(hours, tilts, marker='o')
        plt.title(f'Simulated Optimal Tilt Angle Across the Day\n(Lat: {latitude}, Lon: {longitude})')
        plt.xlabel('Hour (UTC)')
        plt.ylabel('Tilt Angle (Degrees)')
        plt.grid(True)
        plt.show()

    except ValueError:
        print("Invalid input. Please enter numeric values for latitude and longitude.")
