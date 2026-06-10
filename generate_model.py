import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

def main():
    print("Initiating synthetic data generation for Chennai hydrological modeling...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Number of standard samples
    n_samples = 10000
    
    # Features:
    # 1. area_type: 0 to 4 (representing typographic vulnerability: 0=marshland/coastal, 4=elevated ridge)
    area_types = np.random.randint(0, 5, size=n_samples)
    
    # 2. rainfall_rate_mm: 10 to 150 mm/hr
    rainfall_rates = np.random.uniform(10, 150, size=n_samples)
    
    # 3. duration_hours: 1 to 24 hours
    durations = np.random.randint(1, 25, size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'area_type': area_types,
        'rainfall_rate_mm': rainfall_rates,
        'duration_hours': durations
    })
    
    # Calculate total cumulative rainfall
    df['total_rainfall_mm'] = df['rainfall_rate_mm'] * df['duration_hours']
    
    # Probability distribution mapping real-world physical thresholds
    def get_flood_probability(row):
        area_type = row['area_type']
        rate = row['rainfall_rate_mm']
        duration = row['duration_hours']
        total_rain = row['total_rainfall_mm']
        
        # Extreme event thresholds (super heavy rainfall always causes flood risk)
        if total_rain > 320:
            return 0.98 if area_type < 4 else 0.80
            
        if total_rain > 220:
            if area_type == 4:
                return 0.45
            if area_type == 3:
                return 0.75
            return 0.95
            
        # Standard typography thresholds
        # 0: coastal/marsh (floods easily, threshold 60mm)
        # 1: low plain (threshold 100mm)
        # 2: medium plain (threshold 150mm)
        # 3: residential ridge (threshold 220mm)
        # 4: uplands (threshold 300mm)
        thresholds = {0: 60, 1: 100, 2: 150, 3: 220, 4: 300}
        thresh = thresholds[area_type]
        
        if total_rain < thresh * 0.5:
            # Below half threshold, flooding is highly unlikely
            prob = 0.02
        elif total_rain < thresh:
            # Linear transition to threshold
            prob = 0.02 + 0.38 * ((total_rain - thresh * 0.5) / (thresh * 0.5))
        else:
            # Exceeded threshold
            prob = 0.40 + 0.58 * min(1.0, (total_rain - thresh) / thresh)
            
        # High-intensity flash flood (extreme hourly intensity, e.g. > 90mm/hr)
        if rate > 90 and duration >= 2:
            prob = max(prob, 0.70 if area_type >= 3 else 0.95)
            
        return prob
        
    probs = df.apply(get_flood_probability, axis=1)
    
    # Bernoulli trial to get binary labels
    df['flood_occurred'] = np.random.binomial(1, probs)
    
    # --- Inject Specific Historical Scenarios ---
    
    # 2015 Scenario: Widespread intense precipitation. 
    # Average rate: ~23 mm/hr for 21 hours (~480mm total).
    print("Simulating historical 2015 Monsoonal storm events...")
    n_2015 = 200
    area_types_2015 = np.random.randint(0, 5, size=n_2015)
    rates_2015 = np.random.uniform(18, 28, size=n_2015)
    durations_2015 = np.random.randint(18, 25, size=n_2015)
    
    flood_2015 = []
    for at in area_types_2015:
        if at == 4:
            flood_2015.append(np.random.binomial(1, 0.75)) # high elevations mostly flooded due to drainage failure
        else:
            flood_2015.append(1) # other areas completely flooded
            
    df_2015 = pd.DataFrame({
        'area_type': area_types_2015,
        'rainfall_rate_mm': rates_2015,
        'duration_hours': durations_2015,
        'total_rainfall_mm': rates_2015 * durations_2015,
        'flood_occurred': flood_2015
    })
    
    # 2023 Scenario (Cyclone Michaung): Heavy monsoonal rainfall.
    # Average rate: ~15 mm/hr for 22 hours (~330mm total).
    print("Simulating historical 2023 Cyclone Michaung precipitation patterns...")
    n_2023 = 200
    area_types_2023 = np.random.randint(0, 5, size=n_2023)
    rates_2023 = np.random.uniform(12, 18, size=n_2023)
    durations_2023 = np.random.randint(20, 25, size=n_2023)
    
    flood_2023 = []
    for at in area_types_2023:
        if at in [0, 1, 2]:
            flood_2023.append(1) # marsh, lowlands, flatlands flooded
        elif at == 3:
            flood_2023.append(np.random.binomial(1, 0.60)) # moderate flood risk for residential ridges
        else:
            flood_2023.append(np.random.binomial(1, 0.30)) # low risk for uplands
            
    df_2023 = pd.DataFrame({
        'area_type': area_types_2023,
        'rainfall_rate_mm': rates_2023,
        'duration_hours': durations_2023,
        'total_rainfall_mm': rates_2023 * durations_2023,
        'flood_occurred': flood_2023
    })
    
    # Combine standard synthetic data with historical simulations
    df_final = pd.concat([df, df_2015, df_2023], ignore_index=True)
    
    # Shuffle dataset
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select features
    features = ['area_type', 'rainfall_rate_mm', 'duration_hours']
    X = df_final[features]
    y = df_final['flood_occurred']
    
    # Stratified split to maintain class balance in training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # RandomForest training
    print("Training RandomForest Classifier model...")
    clf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE REPORT")
    print("="*40)
    print(f"Dataset Size: {len(df_final)} records")
    print(f"Training Accuracy: {clf.score(X_train, y_train) * 100:.2f}%")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Metrics:")
    print(classification_report(y_test, y_pred))
    print("="*40)
    
    # Save serialized model
    with open('flood_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Successfully serialized model and saved to 'flood_model.pkl'.\n")

if __name__ == '__main__':
    main()
