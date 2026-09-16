"""
=============================================================================
  PREDICTIVE ABORT DETECTION - ROCKET ENGINE TEST STAND SIMULATION
  Teaching Simulation for Rocket Engine Health Monitoring
=============================================================================

This simulation models a liquid rocket engine static-fire test.
The engine has 13 sensor channels:
  - Vibration sensors (accelerometers)      -> high-frequency, fast sensors
  - Chamber pressure                         -> tells if combustion is correct
  - Propellant temperatures                  -> fuel/oxidizer temperatures
  - Mass flow rates                          -> how much fuel/oxidizer flows
  - Turbopump speed (RPM)                    -> rotation speed of pump
  - Thrust (load cell)                       -> how much force engine makes

The goal: PREDICT an abort BEFORE the redline (safety limit) is hit.
Old method: Wait for one sensor to cross a fixed limit (slow, dangerous).
New method: Watch ALL sensors together with a machine learning model (faster!).

=============================================================================
HOW TO RUN:
  python simulation.py

WHAT YOU WILL GET:
  - Simulated sensor data saved as CSV
  - Several graphs showing sensor behavior
  - A trained Random Forest model
  - Comparison: old redline method vs new ML method
=============================================================================
"""

# ============================================================
# STEP 0: IMPORT LIBRARIES
# These are Python tools we need. Like getting your equipment ready.
# ============================================================
import numpy as np                          # For math and numbers
import pandas as pd                         # For tables (like Excel)
import matplotlib
matplotlib.use('Agg')                       # Save to file without needing a screen window
                                            # TIP FOR STUDENTS: Remove the line above to get
                                            #   interactive popup graph windows instead of just files
import matplotlib.pyplot as plt             # For drawing graphs
import matplotlib.gridspec as gridspec      # For arranging multiple graphs
from matplotlib.patches import Patch        # For graph legends
from scipy import signal                    # For signal processing (FFT)
from sklearn.ensemble import RandomForestClassifier  # Our ML model
from sklearn.model_selection import train_test_split # Split data for testing
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler     # Normalize data
import warnings
warnings.filterwarnings('ignore')

# Set a random seed so results are the same every time
np.random.seed(42)

print("=" * 70)
print("  PREDICTIVE ABORT DETECTION - ROCKET ENGINE SIMULATION")
print("=" * 70)
print()


# ============================================================
# STEP 1: SIMULATION SETTINGS
# Think of this like configuring your test before you run it.
# ============================================================

# How long the test runs (seconds)
TEST_DURATION_SECONDS = 120     # 2 minutes

# Sampling rates - how many readings per second each sensor takes
VIBRATION_SAMPLE_RATE = 5000    # 5000 readings/sec (fast sensor)
PROCESS_SAMPLE_RATE   = 100     # 100 readings/sec (slow sensors)

# Feature extraction window - we look at 0.1 second chunks of vibration data
WINDOW_SIZE_SEC = 0.1
WINDOW_SAMPLES  = int(VIBRATION_SAMPLE_RATE * WINDOW_SIZE_SEC)  # = 500 samples

# Number of windows we will create from the test
N_WINDOWS = int(TEST_DURATION_SECONDS / WINDOW_SIZE_SEC)         # = 1200 windows

# When does the fault start? (secretly, for simulation only)
FAULT_START_WINDOW  = 800   # Fault begins at window 800 (80 seconds)
ABORT_WINDOW        = 1050  # Old redline would trigger here (105 seconds)
                             # Our model should predict BEFORE this!

print(f"[CONFIG] Test Duration     : {TEST_DURATION_SECONDS} seconds")
print(f"[CONFIG] Vibration Rate    : {VIBRATION_SAMPLE_RATE} Hz")
print(f"[CONFIG] Process Rate      : {PROCESS_SAMPLE_RATE} Hz")
print(f"[CONFIG] Analysis Windows  : {N_WINDOWS} windows of {WINDOW_SIZE_SEC}s each")
print(f"[CONFIG] Fault starts at   : Window {FAULT_START_WINDOW} ({FAULT_START_WINDOW*WINDOW_SIZE_SEC:.0f}s)")
print(f"[CONFIG] Redline triggers  : Window {ABORT_WINDOW} ({ABORT_WINDOW*WINDOW_SIZE_SEC:.0f}s)")
print()


# ============================================================
# STEP 2: SIMULATE THE ENGINE - Generate Sensor Data
# We are "pretending" to be a rocket engine and creating fake
# but realistic sensor readings. Real engineers use real hardware.
# ============================================================

print("[STEP 2] Simulating engine sensor data...")

def simulate_vibration(n_windows, fault_start, abort_window, window_samples, fs):
    """
    Simulate a vibration sensor on the engine.
    
    Normal operation: random noise (engine always vibrates a little)
    Developing fault: a new frequency appears and grows stronger
    Near abort:       vibration becomes very chaotic and loud
    
    Returns: array of shape (n_windows, window_samples)
              Each row = one time window of vibration readings
    """
    vibration_windows = []
    t_window = np.linspace(0, 1.0 / (fs / window_samples), window_samples)
    
    for w in range(n_windows):
        # ---- Normal background vibration ----
        # Engine always has some base vibration frequencies
        base = (0.5 * np.sin(2 * np.pi * 120 * t_window) +    # Turbopump blade pass
                0.3 * np.sin(2 * np.pi * 340 * t_window) +    # Structural resonance
                0.2 * np.sin(2 * np.pi * 85  * t_window))     # Shaft frequency
        noise = np.random.normal(0, 0.15, window_samples)     # Random noise
        
        # ---- Fault development ----
        if w >= fault_start:
            # How far into the fault are we? (0 = just started, 1 = full abort)
            fault_progress = (w - fault_start) / (abort_window - fault_start)
            fault_progress = min(fault_progress, 1.0)
            
            # A new "bad" frequency appears - like a bearing starting to crack
            fault_freq     = 67.5  # Hz - a sideband frequency (physically meaningful)
            fault_amp      = 1.5 * fault_progress ** 1.5   # Gets stronger over time
            fault_signal   = fault_amp * np.sin(2 * np.pi * fault_freq * t_window)
            
            # Increase in broadband noise (structure getting noisier)
            noise_amp      = 0.15 + 0.6 * fault_progress
            noise          = np.random.normal(0, noise_amp, window_samples)
            
            vibration = base + fault_signal + noise
        else:
            vibration = base + noise
        
        vibration_windows.append(vibration)
    
    return np.array(vibration_windows)   # Shape: (N_WINDOWS, WINDOW_SAMPLES)


def simulate_process_variables(n_windows, fault_start, abort_window):
    """
    Simulate the slow process sensors:
    - Chamber pressure (bar)
    - Fuel temperature (K)
    - Oxidizer temperature (K)
    - Fuel mass flow (kg/s)
    - Oxidizer mass flow (kg/s)
    - Turbopump RPM
    - Thrust (kN)
    
    These change slowly. The fault affects them, but LATER than vibration.
    That's the key lesson: vibration warns us FIRST!
    
    Returns: DataFrame with one row per window
    """
    data = []
    
    # Normal operation setpoints
    P_CHAMBER_NOMINAL   = 180.0   # bar
    T_FUEL_NOMINAL      = 110.0   # K (cryogenic fuel)
    T_OX_NOMINAL        = 90.0    # K (cryogenic oxidizer)
    MDOT_FUEL_NOMINAL   = 45.0    # kg/s
    MDOT_OX_NOMINAL     = 90.0    # kg/s (oxidizer-to-fuel ratio ~ 2)
    RPM_NOMINAL         = 32000   # rpm
    THRUST_NOMINAL      = 950.0   # kN
    
    for w in range(n_windows):
        fault_progress = 0.0
        if w >= fault_start:
            fault_progress = (w - fault_start) / (abort_window - fault_start)
            fault_progress = min(fault_progress, 1.2)  # can exceed 1 near abort
        
        # Slow drift starts much later than vibration (this is the KEY POINT)
        # Process variables only start drifting when fault_progress > 0.4
        slow_fault = max(0.0, (fault_progress - 0.4) / 0.6) ** 2
        
        row = {
            'window': w,
            'time_s': w * WINDOW_SIZE_SEC,
            # Pressure drops as combustion becomes unstable
            'chamber_pressure_bar': (P_CHAMBER_NOMINAL
                                     - 18.0 * slow_fault
                                     + np.random.normal(0, 0.8)),
            # Temperatures rise (turbopump heating up due to friction/cavitation)
            'fuel_temp_K':    (T_FUEL_NOMINAL
                               + 4.0 * slow_fault
                               + np.random.normal(0, 0.5)),
            'ox_temp_K':      (T_OX_NOMINAL
                               + 3.0 * slow_fault
                               + np.random.normal(0, 0.4)),
            # Flow rates change as pump cavitates
            'fuel_flow_kgs':  (MDOT_FUEL_NOMINAL
                               - 3.5 * slow_fault
                               + np.random.normal(0, 0.3)),
            'ox_flow_kgs':    (MDOT_OX_NOMINAL
                               - 6.0 * slow_fault
                               + np.random.normal(0, 0.5)),
            # RPM drops as turbopump degrades
            'turbopump_rpm':  (RPM_NOMINAL
                               - 900 * slow_fault
                               + np.random.normal(0, 50)),
            # Thrust drops as efficiency reduces
            'thrust_kN':      (THRUST_NOMINAL
                               - 60.0 * slow_fault
                               + np.random.normal(0, 3.0)),
            # Ground truth label (only known in simulation, not in real life!)
            'label': 1 if w >= fault_start else 0
        }
        data.append(row)
    
    return pd.DataFrame(data)


# Run the simulations
print("   Generating vibration data (this takes a moment)...")
vibration_data = simulate_vibration(
    N_WINDOWS, FAULT_START_WINDOW, ABORT_WINDOW,
    WINDOW_SAMPLES, VIBRATION_SAMPLE_RATE
)
print(f"   Vibration data shape: {vibration_data.shape}  (windows x samples)")

print("   Generating process variable data...")
process_df = simulate_process_variables(N_WINDOWS, FAULT_START_WINDOW, ABORT_WINDOW)
print(f"   Process data shape: {process_df.shape}  (windows x channels)")
print()


# ============================================================
# STEP 3: FEATURE ENGINEERING
# We cannot feed raw vibration data to ML directly (500 numbers per window
# is too much and too noisy). Instead we SUMMARIZE each window into a few
# meaningful numbers called FEATURES.
#
# This is like summarizing a whole song by saying:
# "It is loud, has a fast tempo, and the bass is strong."
# ============================================================

print("[STEP 3] Extracting features from vibration windows (FFT-based)...")

def extract_vibration_features(window, fs):
    """
    From one window of raw vibration (e.g. 500 samples), compute:
    
    1. RMS - Root Mean Square - "Overall energy / loudness"
    2. Peak amplitude - "Highest spike"
    3. Crest factor - Peak / RMS - "How spiky vs. smooth"
    4. Spectral centroid - "Where is the frequency energy centered?"
    5. Band energy ratio - Energy in 50-100 Hz / total (fault band)
    6. Spectral entropy - "How spread out is the energy?" (disorder)
    
    Returns: dict with 6 feature values for this window
    """
    # Fast Fourier Transform - convert time signal to frequency domain
    freqs = np.fft.rfftfreq(len(window), d=1.0/fs)  # Frequency axis
    fft_amplitudes = np.abs(np.fft.rfft(window))     # How strong each frequency is
    power = fft_amplitudes ** 2                       # Power = amplitude squared
    
    # 1. RMS (Root Mean Square) - overall vibration level
    rms = np.sqrt(np.mean(window ** 2))
    
    # 2. Peak amplitude
    peak = np.max(np.abs(window))
    
    # 3. Crest factor (high value = impulsive/spiky - common in bearing faults)
    crest_factor = peak / (rms + 1e-10)
    
    # 4. Spectral centroid - "center of gravity" of the frequency spectrum
    total_power = np.sum(power) + 1e-10
    spectral_centroid = np.sum(freqs * power) / total_power
    
    # 5. Fault band energy ratio - energy in fault frequency zone (50-100 Hz)
    fault_mask    = (freqs >= 50) & (freqs <= 100)
    fault_energy  = np.sum(power[fault_mask])
    band_ratio    = fault_energy / total_power
    
    # 6. Spectral entropy - how "disordered" is the spectrum
    #    Low entropy = energy concentrated at a few frequencies (healthy engine)
    #    High entropy = energy spread everywhere (damaged engine, more chaotic)
    prob = power / total_power
    prob = prob[prob > 0]  # Remove zeros to avoid log(0)
    spectral_entropy = -np.sum(prob * np.log(prob))
    # Normalize entropy to 0-1 range
    max_entropy = np.log(len(freqs))
    spectral_entropy = spectral_entropy / max_entropy
    
    return {
        'vib_rms':              rms,
        'vib_peak':             peak,
        'vib_crest_factor':     crest_factor,
        'vib_spectral_centroid': spectral_centroid,
        'vib_band_energy_ratio': band_ratio,
        'vib_spectral_entropy':  spectral_entropy,
    }


# Extract features for all windows
all_features = []
for w in range(N_WINDOWS):
    feats = extract_vibration_features(vibration_data[w], VIBRATION_SAMPLE_RATE)
    all_features.append(feats)

vibration_feature_df = pd.DataFrame(all_features)
print(f"   Vibration features shape: {vibration_feature_df.shape}")
print(f"   Features extracted: {list(vibration_feature_df.columns)}")
print()


# ============================================================
# STEP 4: COMBINE FEATURES INTO ONE DATASET
# Join vibration features + process sensor data -> 13 total columns
# This is the "feature vector" that our ML model will learn from.
# ============================================================

print("[STEP 4] Building combined feature dataset...")

# Process variable columns (drop non-feature columns)
process_cols = ['chamber_pressure_bar', 'fuel_temp_K', 'ox_temp_K',
                'fuel_flow_kgs', 'ox_flow_kgs', 'turbopump_rpm', 'thrust_kN']

# Combine everything into one big table
dataset = pd.concat([
    process_df[['window', 'time_s', 'label']].reset_index(drop=True),
    process_df[process_cols].reset_index(drop=True),
    vibration_feature_df.reset_index(drop=True)
], axis=1)

print(f"   Combined dataset shape: {dataset.shape}")
print(f"   Columns: {list(dataset.columns)}")
print(f"   Label distribution: Nominal={sum(dataset.label==0)}, Abort-risk={sum(dataset.label==1)}")
print()

# Save the raw dataset so students can look at it in Excel
dataset.to_csv('simulation_data.csv', index=False)
print("   [SAVED] simulation_data.csv - You can open this in Excel!")
print()


# ============================================================
# STEP 5: PREPARE DATA FOR MACHINE LEARNING
# We split the data into:
#   - TRAINING SET (70%): The model learns patterns from this
#   - TEST SET (30%):     We check if the model learned correctly
#
# IMPORTANT: We shuffle randomly so the model doesn't just memorize order.
# ============================================================

print("[STEP 5] Preparing data for machine learning...")

FEATURE_COLS = process_cols + list(vibration_feature_df.columns)

X = dataset[FEATURE_COLS].values   # Input features (13 columns)
y = dataset['label'].values        # Output labels (0=Normal, 1=Abort-risk)

# Split: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Normalize features - make all numbers similar scale (0 to 1 roughly)
# Important because some features are in bar (pressure) and some in RPM (thousands)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Learn scaling from training data
X_test_scaled  = scaler.transform(X_test)        # Apply same scaling to test data

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples    : {X_test.shape[0]}")
print(f"   Feature count   : {X_train.shape[1]}")
print()


# ============================================================
# STEP 6: TRAIN THE RANDOM FOREST MODEL
# Random Forest = many decision trees, each voting on the answer.
# Like asking 300 different experts: "Is this engine okay?"
# The majority vote wins.
# ============================================================

print("[STEP 6] Training the Random Forest classifier...")
print("   (300 decision trees, like 300 experts voting...)")

model = RandomForestClassifier(
    n_estimators=300,       # 300 trees
    max_depth=8,            # Each tree can be at most 8 levels deep
    min_samples_leaf=5,     # Each leaf node needs at least 5 samples
    class_weight='balanced',# Handle imbalanced data (fewer abort examples)
    random_state=42,
    n_jobs=-1               # Use all CPU cores for speed
)

model.fit(X_train_scaled, y_train)
print("   Training complete!")
print()


# ============================================================
# STEP 7: EVALUATE THE MODEL
# How well did our model learn?
# We use precision, recall, F1, and AUC-ROC to measure performance.
# ============================================================

print("[STEP 7] Evaluating model performance...")
print("-" * 50)

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of "abort-risk"

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Nominal', 'Abort-Risk']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("                 Predicted Nominal | Predicted Abort-Risk")
print(f"  Actual Nominal   :   {cm[0,0]:4d}         |   {cm[0,1]:4d}")
print(f"  Actual Abort-Risk:   {cm[1,0]:4d}         |   {cm[1,1]:4d}")
print()

auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc:.4f}  (1.0 = perfect, 0.5 = random guessing)")
print()


# ============================================================
# STEP 8: FEATURE IMPORTANCE
# Which sensors/features matter most to the model?
# This helps engineers understand WHAT the model is looking at.
# ============================================================

print("[STEP 8] Calculating feature importances...")

importances = model.feature_importances_
feature_names = FEATURE_COLS
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 13 Most Important Features:")
for _, row in fi_df.iterrows():
    bar = '=' * int(row['importance'] * 100)
    print(f"  {row['feature']:<30s}: {row['importance']:.4f}  {bar}")
print()


# ============================================================
# STEP 9: CALCULATE LEAD TIME
# The most important question: HOW EARLY does our model predict
# the abort compared to the old redline system?
# ============================================================

print("[STEP 9] Calculating early warning lead time...")

# Apply model to the FULL time sequence (not shuffled)
X_full        = dataset[FEATURE_COLS].values
X_full_scaled = scaler.transform(X_full)
proba_full    = model.predict_proba(X_full_scaled)[:, 1]

# Define: model triggers when probability crosses 0.5 threshold
ML_THRESHOLD = 0.50

# Find the first window where model predicts abort-risk
ml_trigger_windows = np.where(
    (proba_full > ML_THRESHOLD) & (dataset['window'] > FAULT_START_WINDOW)
)[0]

if len(ml_trigger_windows) > 0:
    ml_trigger_window = ml_trigger_windows[0]
    ml_trigger_time   = ml_trigger_window * WINDOW_SIZE_SEC
else:
    ml_trigger_window = ABORT_WINDOW
    ml_trigger_time   = ABORT_WINDOW * WINDOW_SIZE_SEC

redline_time  = ABORT_WINDOW * WINDOW_SIZE_SEC
lead_time_sec = redline_time - ml_trigger_time

print(f"   Fault actual start  : {FAULT_START_WINDOW * WINDOW_SIZE_SEC:.1f}s (window {FAULT_START_WINDOW})")
print(f"   Redline (old method): {redline_time:.1f}s (window {ABORT_WINDOW})")
print(f"   ML model prediction : {ml_trigger_time:.1f}s (window {ml_trigger_window})")
print(f"   LEAD TIME ADVANTAGE : {lead_time_sec:.1f} seconds earlier than redline!")
print()


# ============================================================
# STEP 10: GENERATE GRAPHS
# Now we make charts so humans can understand what happened.
# "A picture is worth a thousand numbers."
# ============================================================

print("[STEP 10] Generating graphs and charts...")
print("   This will open multiple figure windows.")
print()

# ---- Colors we will use across all plots ----
C_NORMAL    = '#2196F3'   # Blue  = Normal operation
C_FAULT     = '#FF5722'   # Red   = Fault developing
C_ML        = '#4CAF50'   # Green = ML model trigger
C_REDLINE   = '#FF9800'   # Orange = Old redline trigger
C_GRID      = '#303030'   # Dark grid lines


# ========================
# FIGURE 1: Sensor Time Series
# Show how each sensor changes over time during the test
# ========================
fig1, axes = plt.subplots(4, 2, figsize=(16, 14))
fig1.patch.set_facecolor('#1A1A2E')   # Dark background
fig1.suptitle('FIGURE 1: Sensor Time Series During Engine Test\n'
              '(Blue = Normal Phase  |  Red = Fault Developing Phase)',
              fontsize=14, color='white', fontweight='bold')

time_axis = dataset['time_s'].values

plot_configs = [
    ('chamber_pressure_bar', 'Chamber Pressure (bar)',       C_NORMAL, 0, 0),
    ('fuel_temp_K',          'Fuel Temperature (K)',         C_NORMAL, 0, 1),
    ('ox_temp_K',            'Oxidizer Temperature (K)',     C_NORMAL, 1, 0),
    ('fuel_flow_kgs',        'Fuel Mass Flow (kg/s)',        C_NORMAL, 1, 1),
    ('ox_flow_kgs',          'Oxidizer Mass Flow (kg/s)',    C_NORMAL, 2, 0),
    ('turbopump_rpm',        'Turbopump RPM',                C_NORMAL, 2, 1),
    ('thrust_kN',            'Thrust (kN)',                  C_NORMAL, 3, 0),
    ('vib_rms',              'Vibration RMS (g)',            '#9C27B0', 3, 1),
]

fault_time    = FAULT_START_WINDOW * WINDOW_SIZE_SEC
redline_time2 = ABORT_WINDOW * WINDOW_SIZE_SEC

for col, title, color, row, col_idx in plot_configs:
    ax = axes[row][col_idx]
    ax.set_facecolor('#16213E')
    
    values = dataset[col].values
    # Color the line differently in the fault zone
    normal_mask = time_axis <= fault_time
    fault_mask  = time_axis > fault_time
    
    ax.plot(time_axis[normal_mask], values[normal_mask],
            color=C_NORMAL, linewidth=1.2, label='Normal')
    ax.plot(time_axis[fault_mask],  values[fault_mask],
            color=C_FAULT,  linewidth=1.2, label='Fault zone')
    
    # Draw vertical lines for events
    ax.axvline(fault_time,    color='yellow',  linestyle='--', linewidth=1.5,
               alpha=0.8, label='Fault begins')
    ax.axvline(ml_trigger_time, color=C_ML,   linestyle='--', linewidth=2,
               alpha=0.9, label='ML trigger')
    ax.axvline(redline_time2, color=C_REDLINE, linestyle=':',  linewidth=2,
               alpha=0.9, label='Redline trigger')
    
    ax.set_title(title, color='white', fontsize=9, fontweight='bold')
    ax.set_xlabel('Time (s)', color='#AAAAAA', fontsize=8)
    ax.tick_params(colors='#AAAAAA', labelsize=7)
    ax.spines[['bottom','top','left','right']].set_color('#444444')
    ax.grid(True, alpha=0.2, color=C_GRID)

# Remove the empty 8th plot area legend and put one shared legend
axes[3][1].legend(loc='lower left', fontsize=7, facecolor='#1A1A2E',
                  labelcolor='white', framealpha=0.8)

plt.tight_layout()
plt.savefig('fig1_sensor_timeseries.png', dpi=150, bbox_inches='tight',
            facecolor='#1A1A2E')
print("   [SAVED] fig1_sensor_timeseries.png")


# ========================
# FIGURE 2: Vibration Feature Trends
# Show how the 6 vibration features change over time
# ========================
fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12))
fig2.patch.set_facecolor('#1A1A2E')
fig2.suptitle('FIGURE 2: Vibration Feature Trends\n'
              'These are the numbers extracted from raw vibration data (FFT features)',
              fontsize=13, color='white', fontweight='bold')

vib_features_plot = [
    ('vib_rms',               'Vibration RMS\n(Overall energy level)',   0, 0),
    ('vib_peak',              'Vibration Peak\n(Maximum amplitude)',      0, 1),
    ('vib_crest_factor',      'Crest Factor\n(Spikiness of vibration)',   1, 0),
    ('vib_spectral_centroid', 'Spectral Centroid (Hz)\n(Where is energy concentrated)', 1, 1),
    ('vib_band_energy_ratio', 'Fault Band Energy Ratio\n(Energy at 50-100 Hz)', 2, 0),
    ('vib_spectral_entropy',  'Spectral Entropy\n(Disorder level of vibration)', 2, 1),
]

for feat, title, row, col_idx in vib_features_plot:
    ax = axes2[row][col_idx]
    ax.set_facecolor('#16213E')
    values = dataset[feat].values
    
    ax.fill_between(time_axis,
                    np.zeros_like(values),
                    values,
                    where=(time_axis <= fault_time),
                    color=C_NORMAL, alpha=0.4, label='Normal')
    ax.fill_between(time_axis,
                    np.zeros_like(values),
                    values,
                    where=(time_axis > fault_time),
                    color=C_FAULT, alpha=0.4, label='Fault zone')
    ax.plot(time_axis, values, color='white', linewidth=0.8, alpha=0.6)
    
    ax.axvline(fault_time,    color='yellow',   linestyle='--', linewidth=1.5)
    ax.axvline(ml_trigger_time, color=C_ML,     linestyle='--', linewidth=2)
    ax.axvline(redline_time2, color=C_REDLINE,  linestyle=':',  linewidth=2)
    
    ax.set_title(title, color='white', fontsize=9, fontweight='bold')
    ax.set_xlabel('Time (s)', color='#AAAAAA', fontsize=8)
    ax.tick_params(colors='#AAAAAA', labelsize=7)
    ax.spines[['bottom','top','left','right']].set_color('#444444')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('fig2_vibration_features.png', dpi=150, bbox_inches='tight',
            facecolor='#1A1A2E')
print("   [SAVED] fig2_vibration_features.png")


# ========================
# FIGURE 3: FFT Spectra Comparison
# Show what the vibration frequency content looks like:
# one window from Normal operation, one from Fault zone
# ========================
fig3, (ax_norm, ax_fault, ax_diff) = plt.subplots(1, 3, figsize=(16, 5))
fig3.patch.set_facecolor('#1A1A2E')
fig3.suptitle('FIGURE 3: Vibration Frequency Spectrum (FFT)\n'
              'Comparing Normal vs Fault: The fault creates a new frequency spike!',
              fontsize=13, color='white', fontweight='bold')

# Pick one Normal window and one Fault window to visualize
normal_window_idx = 300
fault_window_idx  = 1000

def compute_fft_spectrum(window, fs):
    freqs = np.fft.rfftfreq(len(window), d=1.0/fs)
    amps  = np.abs(np.fft.rfft(window))
    # Only show up to 500 Hz (the interesting range)
    mask  = freqs <= 500
    return freqs[mask], amps[mask]

f_norm,  a_norm  = compute_fft_spectrum(vibration_data[normal_window_idx],
                                         VIBRATION_SAMPLE_RATE)
f_fault, a_fault = compute_fft_spectrum(vibration_data[fault_window_idx],
                                         VIBRATION_SAMPLE_RATE)

for ax, freqs, amps, title, color, t_label in [
    (ax_norm,  f_norm,  a_norm,  f'Normal Window\n(t = {normal_window_idx*WINDOW_SIZE_SEC:.0f}s)',  C_NORMAL,  'Normal'),
    (ax_fault, f_fault, a_fault, f'Fault Window\n(t = {fault_window_idx*WINDOW_SIZE_SEC:.0f}s)',   C_FAULT,   'Fault'),
]:
    ax.set_facecolor('#16213E')
    ax.fill_between(freqs, amps, color=color, alpha=0.5)
    ax.plot(freqs, amps, color=color, linewidth=1)
    ax.axvline(67.5, color='yellow', linestyle='--', linewidth=2,
               label='Fault freq (67.5 Hz)')
    ax.axvline(85,   color='cyan',   linestyle=':', linewidth=1.5,
               label='Shaft freq (85 Hz)')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', color='#AAAAAA')
    ax.set_ylabel('Amplitude (g)', color='#AAAAAA')
    ax.tick_params(colors='#AAAAAA')
    ax.spines[['bottom','top','left','right']].set_color('#444444')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, facecolor='#1A1A2E', labelcolor='white')

# Difference spectrum
ax_diff.set_facecolor('#16213E')
diff_amps = a_fault - a_norm
ax_diff.fill_between(f_norm, diff_amps, where=(diff_amps > 0),
                     color=C_FAULT, alpha=0.6, label='Increased')
ax_diff.fill_between(f_norm, diff_amps, where=(diff_amps < 0),
                     color=C_NORMAL, alpha=0.6, label='Decreased')
ax_diff.axhline(0, color='white', linewidth=0.8)
ax_diff.axvline(67.5, color='yellow', linestyle='--', linewidth=2, label='Fault freq')
ax_diff.set_title('Difference (Fault - Normal)\nThe spike at 67.5 Hz = fault signature!',
                  color='white', fontsize=10, fontweight='bold')
ax_diff.set_xlabel('Frequency (Hz)', color='#AAAAAA')
ax_diff.set_ylabel('Amplitude Difference (g)', color='#AAAAAA')
ax_diff.tick_params(colors='#AAAAAA')
ax_diff.spines[['bottom','top','left','right']].set_color('#444444')
ax_diff.grid(True, alpha=0.2)
ax_diff.legend(fontsize=7, facecolor='#1A1A2E', labelcolor='white')

plt.tight_layout()
plt.savefig('fig3_fft_spectra.png', dpi=150, bbox_inches='tight',
            facecolor='#1A1A2E')
print("   [SAVED] fig3_fft_spectra.png")


# ========================
# FIGURE 4: ML Model - Probability Timeline
# The most important graph: shows the model's confidence over time
# and compares it to the old redline trigger
# ========================
fig4, axes4 = plt.subplots(2, 1, figsize=(14, 8))
fig4.patch.set_facecolor('#1A1A2E')
fig4.suptitle('FIGURE 4: ML Model Prediction vs Old Redline Method\n'
              'GREEN = ML model triggers early   |   ORANGE = Old redline triggers late',
              fontsize=13, color='white', fontweight='bold')

# Top plot: probability of "abort-risk" over time
ax_top = axes4[0]
ax_top.set_facecolor('#16213E')

ax_top.fill_between(time_axis, proba_full, alpha=0.3, color=C_FAULT)
ax_top.plot(time_axis, proba_full, color=C_FAULT, linewidth=1.5,
            label='Abort-Risk Probability')
ax_top.axhline(ML_THRESHOLD, color='white', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Decision threshold ({ML_THRESHOLD})')

# Shade the regions
ax_top.axvspan(0, fault_time, alpha=0.1, color=C_NORMAL, label='Normal zone')
ax_top.axvspan(fault_time, time_axis[-1], alpha=0.1, color=C_FAULT,
               label='Fault zone')

# Event lines
ax_top.axvline(fault_time, color='yellow', linestyle='--',
               linewidth=2, label='Fault begins (hidden in real life)')
ax_top.axvline(ml_trigger_time, color=C_ML, linestyle='-',
               linewidth=3, alpha=0.9, label=f'ML triggers at {ml_trigger_time:.1f}s')
ax_top.axvline(redline_time2, color=C_REDLINE, linestyle='-',
               linewidth=3, alpha=0.9, label=f'Redline triggers at {redline_time2:.1f}s')

ax_top.set_ylabel('Abort-Risk Probability', color='white', fontsize=10)
ax_top.set_xlabel('Time (s)', color='#AAAAAA')
ax_top.tick_params(colors='#AAAAAA')
ax_top.spines[['bottom','top','left','right']].set_color('#444444')
ax_top.grid(True, alpha=0.2)
ax_top.set_ylim(-0.05, 1.1)
ax_top.legend(loc='upper left', fontsize=8, facecolor='#1A1A2E',
              labelcolor='white', ncol=2)

# Annotate lead time
ax_top.annotate(
    f'LEAD TIME: {lead_time_sec:.1f}s earlier!',
    xy=(ml_trigger_time, 0.55),
    xytext=(ml_trigger_time - 25, 0.75),
    arrowprops=dict(arrowstyle='->', color='white', lw=2),
    fontsize=11, color='white', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor=C_ML, alpha=0.8)
)

# Bottom plot: Thrust and Pressure together (to show redline behavior)
ax_bot = axes4[1]
ax_bot.set_facecolor('#16213E')

thrust_norm = dataset['thrust_kN'].values / dataset['thrust_kN'].values.max()
pressure_norm = dataset['chamber_pressure_bar'].values / dataset['chamber_pressure_bar'].values.max()

ax_bot.plot(time_axis, thrust_norm,   color='#9C27B0', linewidth=1.5,
            label='Thrust (normalized)')
ax_bot.plot(time_axis, pressure_norm, color='#00BCD4', linewidth=1.5,
            label='Chamber Pressure (normalized)')
ax_bot.axhline(0.90, color=C_REDLINE, linestyle=':', linewidth=2,
               alpha=0.8, label='Redline limit (90% of nominal)')

ax_bot.axvline(fault_time,      color='yellow',  linestyle='--', linewidth=2)
ax_bot.axvline(ml_trigger_time, color=C_ML,      linestyle='-',  linewidth=3)
ax_bot.axvline(redline_time2,   color=C_REDLINE, linestyle='-',  linewidth=3)

ax_bot.set_ylabel('Normalized Sensor Value', color='white', fontsize=10)
ax_bot.set_xlabel('Time (s)', color='#AAAAAA')
ax_bot.tick_params(colors='#AAAAAA')
ax_bot.spines[['bottom','top','left','right']].set_color('#444444')
ax_bot.grid(True, alpha=0.2)
ax_bot.legend(loc='lower left', fontsize=8, facecolor='#1A1A2E',
              labelcolor='white')

plt.tight_layout()
plt.savefig('fig4_ml_vs_redline.png', dpi=150, bbox_inches='tight',
            facecolor='#1A1A2E')
print("   [SAVED] fig4_ml_vs_redline.png")


# ========================
# FIGURE 5: Feature Importance Bar Chart
# Which sensors/features did the model use most?
# ========================
fig5, ax5 = plt.subplots(figsize=(12, 7))
fig5.patch.set_facecolor('#1A1A2E')
ax5.set_facecolor('#16213E')

colors_fi = []
for f in fi_df['feature']:
    if 'vib' in f:
        colors_fi.append('#9C27B0')  # Purple = vibration feature
    else:
        colors_fi.append('#2196F3')  # Blue = process sensor

bars = ax5.barh(fi_df['feature'], fi_df['importance'],
                color=colors_fi, edgecolor='#555555', height=0.6)

for bar, val in zip(bars, fi_df['importance']):
    ax5.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', ha='left', color='white', fontsize=8)

ax5.set_xlabel('Feature Importance (higher = model uses this more)', color='white',
               fontsize=10)
ax5.set_title('FIGURE 5: Feature Importance - What Does the Model Look At?\n'
              'Purple = Vibration Features  |  Blue = Process Sensor Features',
              color='white', fontsize=12, fontweight='bold')
ax5.tick_params(colors='white', labelsize=9)
ax5.spines[['bottom','top','left','right']].set_color('#444444')
ax5.grid(True, axis='x', alpha=0.2)
ax5.set_xlim(0, fi_df['importance'].max() * 1.25)

legend_elements = [
    Patch(facecolor='#9C27B0', label='Vibration Feature (from FFT)'),
    Patch(facecolor='#2196F3', label='Process Sensor (pressure, temp, flow...)')
]
ax5.legend(handles=legend_elements, loc='lower right', facecolor='#1A1A2E',
           labelcolor='white', fontsize=9)

plt.tight_layout()
plt.savefig('fig5_feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='#1A1A2E')
print("   [SAVED] fig5_feature_importance.png")


# ========================
# FIGURE 6: Confusion Matrix and ROC Curve
# Professional ML evaluation metrics
# ========================
fig6, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(13, 6))
fig6.patch.set_facecolor('#1A1A2E')
fig6.suptitle('FIGURE 6: Model Performance Metrics',
              fontsize=14, color='white', fontweight='bold')

# ---- Confusion Matrix ----
ax_cm.set_facecolor('#16213E')
im = ax_cm.imshow(cm, cmap='Blues', aspect='auto')
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['Predicted\nNominal', 'Predicted\nAbort-Risk'],
                      color='white', fontsize=10)
ax_cm.set_yticklabels(['Actual\nNominal', 'Actual\nAbort-Risk'],
                      color='white', fontsize=10)
ax_cm.set_title('Confusion Matrix', color='white', fontsize=12, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, str(cm[i, j]),
                   ha='center', va='center', fontsize=24,
                   fontweight='bold',
                   color='white' if cm[i, j] < cm.max() * 0.7 else 'black')

# Label meanings
ax_cm.text(0, 0, '\n\n\nTrue Negative\n(Correct: Normal)',
           ha='center', va='center', fontsize=7, color='#AAAAAA')
ax_cm.text(1, 1, '\n\n\nTrue Positive\n(Correct: Abort)',
           ha='center', va='center', fontsize=7, color='#AAAAAA')
ax_cm.text(1, 0, '\n\n\nFalse Positive\n(False Alarm)',
           ha='center', va='center', fontsize=7, color='#FF9800')
ax_cm.text(0, 1, '\n\n\nFalse Negative\n(MISSED ABORT!)',
           ha='center', va='center', fontsize=7, color='#F44336')

# ---- ROC Curve ----
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
ax_roc.set_facecolor('#16213E')
ax_roc.plot(fpr, tpr, color=C_ML, linewidth=2.5,
            label=f'Random Forest (AUC = {auc:.3f})')
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
            label='Random Guessing (AUC = 0.5)')
ax_roc.fill_between(fpr, tpr, alpha=0.2, color=C_ML)
ax_roc.set_xlabel('False Positive Rate (False Alarms)', color='white', fontsize=10)
ax_roc.set_ylabel('True Positive Rate (Correctly Caught)', color='white', fontsize=10)
ax_roc.set_title('ROC Curve\n(Higher = better model)', color='white',
                 fontsize=12, fontweight='bold')
ax_roc.tick_params(colors='#AAAAAA')
ax_roc.spines[['bottom','top','left','right']].set_color('#444444')
ax_roc.grid(True, alpha=0.2)
ax_roc.legend(fontsize=9, facecolor='#1A1A2E', labelcolor='white')
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig('fig6_model_evaluation.png', dpi=150, bbox_inches='tight',
            facecolor='#1A1A2E')
print("   [SAVED] fig6_model_evaluation.png")
print()


# ========================
# FIGURE 7: The Complete Story - Summary Dashboard
# One combined view showing everything
# ========================
fig7 = plt.figure(figsize=(18, 12))
fig7.patch.set_facecolor('#0D1117')
fig7.suptitle(
    'PREDICTIVE ABORT DETECTION — SUMMARY DASHBOARD\n'
    'Rocket Engine Static-Fire Simulation Results',
    fontsize=16, color='white', fontweight='bold', y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig7, hspace=0.45, wspace=0.35)

# ML Probability (top, full width)
ax_prob = fig7.add_subplot(gs[0, :])
ax_prob.set_facecolor('#161B22')
ax_prob.fill_between(time_axis, proba_full, color=C_FAULT, alpha=0.3)
ax_prob.plot(time_axis, proba_full, color=C_FAULT, linewidth=2)
ax_prob.axhline(ML_THRESHOLD, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
ax_prob.axvline(fault_time, color='yellow', linestyle='--', linewidth=1.5)
ax_prob.axvline(ml_trigger_time, color=C_ML, linestyle='-', linewidth=3)
ax_prob.axvline(redline_time2, color=C_REDLINE, linestyle='-', linewidth=3)
ax_prob.axvspan(0, fault_time, alpha=0.05, color='blue')
ax_prob.axvspan(fault_time, ml_trigger_time, alpha=0.05, color='yellow')
ax_prob.axvspan(ml_trigger_time, redline_time2, alpha=0.1, color=C_ML)
ax_prob.set_title('Abort-Risk Probability Over Time', color='white', fontsize=11)
ax_prob.set_ylabel('P(Abort-Risk)', color='#AAAAAA')
ax_prob.set_xlabel('Time (s)', color='#AAAAAA')
ax_prob.tick_params(colors='#AAAAAA', labelsize=8)
ax_prob.set_ylim(-0.05, 1.1)
ax_prob.grid(True, alpha=0.15)
ax_prob.spines[['bottom','top','left','right']].set_color('#444444')

# Add text annotations
ax_prob.text(fault_time - 5, 0.9, 'Fault\nstarts', color='yellow',
             fontsize=8, ha='right', fontweight='bold')
ax_prob.text(ml_trigger_time + 1, 0.9, f'ML: {ml_trigger_time:.0f}s',
             color=C_ML, fontsize=8, ha='left', fontweight='bold')
ax_prob.text(redline_time2 + 1, 0.9, f'Redline:\n{redline_time2:.0f}s',
             color=C_REDLINE, fontsize=8, ha='left', fontweight='bold')

# Middle row: 3 process sensors
for idx, (col, title) in enumerate([
    ('chamber_pressure_bar', 'Chamber\nPressure (bar)'),
    ('turbopump_rpm',        'Turbopump\nRPM'),
    ('thrust_kN',            'Thrust\n(kN)'),
]):
    ax = fig7.add_subplot(gs[1, idx])
    ax.set_facecolor('#161B22')
    values = dataset[col].values
    ax.plot(time_axis[time_axis <= fault_time],
            values[time_axis <= fault_time],
            color=C_NORMAL, linewidth=1.2)
    ax.plot(time_axis[time_axis > fault_time],
            values[time_axis > fault_time],
            color=C_FAULT, linewidth=1.2)
    ax.axvline(ml_trigger_time, color=C_ML, linestyle='-', linewidth=2)
    ax.axvline(redline_time2,   color=C_REDLINE, linestyle='-', linewidth=2)
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel('Time (s)', color='#AAAAAA', fontsize=7)
    ax.tick_params(colors='#AAAAAA', labelsize=7)
    ax.grid(True, alpha=0.15)
    ax.spines[['bottom','top','left','right']].set_color('#444444')

# Bottom row: 3 vibration features
for idx, (col, title) in enumerate([
    ('vib_rms',              'Vibration\nRMS'),
    ('vib_band_energy_ratio','Fault Band\nEnergy Ratio'),
    ('vib_spectral_entropy', 'Spectral\nEntropy'),
]):
    ax = fig7.add_subplot(gs[2, idx])
    ax.set_facecolor('#161B22')
    values = dataset[col].values
    ax.plot(time_axis[time_axis <= fault_time],
            values[time_axis <= fault_time],
            color='#9C27B0', linewidth=1.2)
    ax.plot(time_axis[time_axis > fault_time],
            values[time_axis > fault_time],
            color=C_FAULT, linewidth=1.2)
    ax.axvline(ml_trigger_time, color=C_ML, linestyle='-', linewidth=2)
    ax.axvline(redline_time2,   color=C_REDLINE, linestyle='-', linewidth=2)
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel('Time (s)', color='#AAAAAA', fontsize=7)
    ax.tick_params(colors='#AAAAAA', labelsize=7)
    ax.grid(True, alpha=0.15)
    ax.spines[['bottom','top','left','right']].set_color('#444444')

plt.savefig('fig7_summary_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
print("   [SAVED] fig7_summary_dashboard.png")
print()


# ============================================================
# FINAL SUMMARY
# Print all results clearly at the end
# ============================================================

print("=" * 70)
print("  SIMULATION COMPLETE — RESULTS SUMMARY")
print("=" * 70)
print()
print(f"  Test Duration          : {TEST_DURATION_SECONDS}s | {N_WINDOWS} analysis windows")
print(f"  Total features used    : {len(FEATURE_COLS)}  (6 vibration + 7 process)")
print(f"  Training samples       : {X_train.shape[0]}")
print(f"  Test samples           : {X_test.shape[0]}")
print()
print(f"  MODEL ACCURACY (AUC-ROC): {auc:.4f} / 1.0000")
print()
print(f"  LEAD TIME COMPARISON:")
print(f"    Old redline method   : Triggers at {redline_time2:.1f}s")
print(f"    ML model method      : Triggers at {ml_trigger_time:.1f}s")
print(f"    Lead time advantage  : {lead_time_sec:.1f} SECONDS EARLIER")
print()
print("  OUTPUT FILES:")
print("    simulation_data.csv          - Raw data table (open in Excel)")
print("    fig1_sensor_timeseries.png   - All sensor trends over time")
print("    fig2_vibration_features.png  - Vibration feature trends")
print("    fig3_fft_spectra.png         - Frequency spectrum (FFT) comparison")
print("    fig4_ml_vs_redline.png       - ML vs Redline comparison (KEY FIGURE)")
print("    fig5_feature_importance.png  - What did the model use most?")
print("    fig6_model_evaluation.png    - Confusion matrix + ROC curve")
print("    fig7_summary_dashboard.png   - Complete summary dashboard")
print()
print("  Run 'python simulation.py' again to regenerate all results.")
print("=" * 70)

# NOTE: Graphs are saved as PNG files. Open them in your image viewer.
# To get interactive popup windows, remove the 'matplotlib.use(Agg)' line at the top of the file.
print()
print("  Open the PNG files in your image viewer to see the graphs!")
print("  Or remove 'matplotlib.use(Agg)' at the top to get popup windows.")
