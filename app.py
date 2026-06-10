import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', template_folder='.')

# Geological database mapping Chennai zones & neighborhoods to typographic codes and metrics
# area_type: 0 (coastal/marsh), 1 (lowland plain), 2 (mid plain), 3 (residential ridge), 4 (uplands)
ZONES_DB = {
    "Tiruvottiyur": {
        "zone_id": 1, 
        "area_type": 0, 
        "elevation_m": 2.1, 
        "description": "Coastal Lowland - highly vulnerable to sea-level surges and river backwater.",
        "drainage_rate_mm_hr": 2.0
    },
    "Manali": {
        "zone_id": 2, 
        "area_type": 1, 
        "elevation_m": 4.5, 
        "description": "Industrial Plain - low elevation near Kosasthalaiyar river basin drainage.",
        "drainage_rate_mm_hr": 3.5
    },
    "Madhavaram": {
        "zone_id": 3, 
        "area_type": 4, 
        "elevation_m": 9.2, 
        "description": "Elevated Uplands - high natural elevation with good soil drainage.",
        "drainage_rate_mm_hr": 15.0
    },
    "Tondiarpet": {
        "zone_id": 4, 
        "area_type": 1, 
        "elevation_m": 3.2, 
        "description": "Coastal Flatland - dense urban area with restricted stormwater slope.",
        "drainage_rate_mm_hr": 3.0
    },
    "Royapuram": {
        "zone_id": 4, 
        "area_type": 1, 
        "elevation_m": 3.0, 
        "description": "Coastal Urban Fringe - low-lying flat terrain adjacent to harbor areas.",
        "drainage_rate_mm_hr": 3.0
    },
    "Thiru-Vi-Ka Nagar": {
        "zone_id": 5, 
        "area_type": 2, 
        "elevation_m": 5.8, 
        "description": "Central Flat Plain - highly paved, prone to waterlogging during monsoon peaks.",
        "drainage_rate_mm_hr": 5.5
    },
    "Ambattur": {
        "zone_id": 6, 
        "area_type": 2, 
        "elevation_m": 7.5, 
        "description": "Lake Basin Plain - surrounded by multiple lakes; vulnerable to industrial runoff overflow.",
        "drainage_rate_mm_hr": 6.0
    },
    "Anna Nagar": {
        "zone_id": 7, 
        "area_type": 3, 
        "elevation_m": 8.0, 
        "description": "Medium Elevation Plain - robust storm drain network but local depression risks.",
        "drainage_rate_mm_hr": 10.0
    },
    "Teynampet": {
        "zone_id": 8, 
        "area_type": 3, 
        "elevation_m": 7.2, 
        "description": "Medium Elevation Central - urban commercial core; drains into central canal networks.",
        "drainage_rate_mm_hr": 9.0
    },
    "Kodambakkam": {
        "zone_id": 9, 
        "area_type": 2, 
        "elevation_m": 6.1, 
        "description": "Central Dense Flatland - concrete-heavy ground, increasing flash run-off.",
        "drainage_rate_mm_hr": 5.0
    },
    "Nungambakkam": {
        "zone_id": 9, 
        "area_type": 2, 
        "elevation_m": 6.0, 
        "description": "Central Basin - historically a lakebed; high localized water retention.",
        "drainage_rate_mm_hr": 5.0
    },
    "Mylapore": {
        "zone_id": 9, 
        "area_type": 1, 
        "elevation_m": 4.2, 
        "description": "Coastal Lowland Plain - historic residential belt with aged drainage outlets.",
        "drainage_rate_mm_hr": 4.0
    },
    "Egmore": {
        "zone_id": 9, 
        "area_type": 1, 
        "elevation_m": 3.8, 
        "description": "Cooum River Basin - low-lying area; vulnerable to river backflows during dams discharge.",
        "drainage_rate_mm_hr": 3.5
    },
    "Valasaravakkam": {
        "zone_id": 10, 
        "area_type": 1, 
        "elevation_m": 5.0, 
        "description": "Low-lying Plain - flat residential development, highly dependent on micro-canals.",
        "drainage_rate_mm_hr": 4.0
    },
    "Porur": {
        "zone_id": 10, 
        "area_type": 1, 
        "elevation_m": 4.8, 
        "description": "Lake Fringe Lowland - located directly in the overflow pathways of Porur lake.",
        "drainage_rate_mm_hr": 3.8
    },
    "Alandur": {
        "zone_id": 11, 
        "area_type": 3, 
        "elevation_m": 12.5, 
        "description": "Elevated Plain - decent natural slopes draining toward the Adyar river.",
        "drainage_rate_mm_hr": 11.0
    },
    "Guindy": {
        "zone_id": 11, 
        "area_type": 4, 
        "elevation_m": 15.0, 
        "description": "Hilly Ridge - high elevation with forest cover, excellent natural percolation.",
        "drainage_rate_mm_hr": 18.0
    },
    "Saidapet": {
        "zone_id": 11, 
        "area_type": 1, 
        "elevation_m": 4.0, 
        "description": "Adyar River Plain - low-lying river bank; highly flooded when upstream reservoirs release water.",
        "drainage_rate_mm_hr": 3.5
    },
    "Adyar": {
        "zone_id": 12, 
        "area_type": 3, 
        "elevation_m": 6.5, 
        "description": "Estuary Plain - flatland near Adyar river mouth; vulnerable to high-tide lockage.",
        "drainage_rate_mm_hr": 8.0
    },
    "Velachery": {
        "zone_id": 12, 
        "area_type": 0, 
        "elevation_m": 2.0, 
        "description": "Marshland Fringe - former lake/marsh basin; extremely vulnerable to massive water logging.",
        "drainage_rate_mm_hr": 1.5
    },
    "Perungudi": {
        "zone_id": 13, 
        "area_type": 0, 
        "elevation_m": 1.8, 
        "description": "Marshland Flatland - high water table bordering Pallikaranai marshland.",
        "drainage_rate_mm_hr": 1.8
    },
    "Sholinganallur": {
        "zone_id": 14, 
        "area_type": 0, 
        "elevation_m": 1.5, 
        "description": "Coastal Marshland - low-elevation flat terrain bordering Buckingham canal and marsh corridor.",
        "drainage_rate_mm_hr": 1.5
    },
    "Tambaram / Chromepet": {
        "zone_id": 14, 
        "area_type": 4, 
        "elevation_m": 22.0, 
        "description": "Southern Elevated Ridges - hilly topography; flooding is restricted to narrow natural drains.",
        "drainage_rate_mm_hr": 16.0
    }
}

# Load ML Model
MODEL_PATH = 'flood_model.pkl'
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Machine Learning Random Forest model loaded successfully.")
    else:
        print(f"Warning: '{MODEL_PATH}' not found. Run 'generate_model.py' to train and save the model.")
except Exception as e:
    print(f"Critical error loading model: {e}")

@app.route('/')
def home():
    """Serve the single-page frontend application dashboard."""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Expose POST endpoint for flood risk classification and hydrological simulation details."""
    global model
    
    # Reload model if it failed to load at startup (safeguard)
    if model is None and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Model load failed: {str(e)}'}), 500
            
    if model is None:
        return jsonify({'status': 'error', 'message': 'Prediction model binary is not initialized.'}), 503
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Empty payload.'}), 400
            
        area_name = data.get('area_name')
        rainfall = data.get('rainfall')
        duration = data.get('duration')
        
        # Validation checks
        if not area_name or rainfall is None or duration is None:
            return jsonify({'status': 'error', 'message': 'Missing fields. Required: area_name, rainfall, duration'}), 400
            
        try:
            rainfall = float(rainfall)
            duration = float(duration)
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Rainfall and duration must be numerical.'}), 400
            
        # Check area database
        if area_name not in ZONES_DB:
            return jsonify({'status': 'error', 'message': f"Area '{area_name}' not mapped in zone database."}), 404
            
        zone_info = ZONES_DB[area_name]
        area_type = zone_info['area_type']
        
        # Prepare input features dataframe with matching columns
        input_features = pd.DataFrame([{
            'area_type': area_type,
            'rainfall_rate_mm': rainfall,
            'duration_hours': duration
        }])
        
        # ML Inference
        prediction_class = int(model.predict(input_features)[0])
        prediction_prob = float(model.predict_proba(input_features)[0][1])
        
        # Hydrological Calculations
        cumulative_rainfall = rainfall * duration
        drainage_rate = zone_info['drainage_rate_mm_hr']
        
        # Water accumulation: total rainfall minus soil drainage over the duration
        net_accumulation = max(0.0, cumulative_rainfall - (drainage_rate * duration))
        
        # Estimate drainage clearing time
        if net_accumulation > 0:
            drainage_time = round(net_accumulation / drainage_rate, 1)
        else:
            drainage_time = 0.0
            
        # Determine risk tag and narrative
        if prediction_prob >= 0.75:
            risk_level = "HIGH RISK"
        elif prediction_prob >= 0.35:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "SAFE"
            
        # Historical context mapping
        historical_reference = "Standard monsoonal precipitation."
        if cumulative_rainfall >= 350:
            historical_reference = "Approaching the catastrophic volume of the December 2015 historical deluge. Immediate emergency standby required."
        elif cumulative_rainfall >= 250:
            historical_reference = "Simulates conditions similar to the 2023 Cyclone Michaung waterlogging events. Extensive suburban inundation likely."
        elif cumulative_rainfall >= 120 and area_type in [0, 1]:
            historical_reference = "Exceeds saturation threshold for low-lying coastal zones. Typical local inundation points active."
            
        # Recommendations
        recommendations = []
        if risk_level == "HIGH RISK":
            recommendations = [
                "Evacuate ground floor levels in designated zone immediately.",
                "Halt non-essential transit across sub-surface channels.",
                "Ensure emergency backup power & critical medication stocks are prepared.",
                "Keep local emergency disaster rescue hotlines (1913) active."
            ]
        elif risk_level == "MODERATE RISK":
            recommendations = [
                "Monitor official GCC updates regarding reservoir gate discharge.",
                "Clear nearby micro-drain catchments from plastic blockage.",
                "Park vehicles in elevated zones or multi-story structures.",
                "Prepare basic emergency kit (potable water, torches, powerbanks)."
            ]
        else:
            recommendations = [
                "Standard monsoonal flow. Clear leaves from local storm sewer inlets.",
                "Maintain default situational awareness of seasonal rainfall reports.",
                "Check roof outlets for seamless terrace runoff drainage."
            ]
            
        return jsonify({
            'status': 'success',
            'data': {
                'area_name': area_name,
                'zone_id': zone_info['zone_id'],
                'area_type': area_type,
                'elevation_m': zone_info['elevation_m'],
                'geological_summary': zone_info['description'],
                'rainfall_rate_mm_hr': rainfall,
                'duration_hours': duration,
                'cumulative_rainfall_mm': round(cumulative_rainfall, 1),
                'drainage_rate_mm_hr': drainage_rate,
                'net_accumulation_mm': round(net_accumulation, 1),
                'drainage_time_hours': drainage_time,
                'prediction': prediction_class,
                'probability': round(prediction_prob, 4),
                'risk_level': risk_level,
                'historical_reference': historical_reference,
                'recommendations': recommendations
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Server processed error: {str(e)}'}), 500

if __name__ == '__main__':
    # Get port from environment (assigned by cloud host) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Listen on all interface routes for web accessibility
    app.run(host='0.0.0.0', port=port, debug=True)
