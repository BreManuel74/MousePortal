import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import time
import serial

def connect_to_arduino(port, baud_rate):
    """Establish a connection to the Arduino."""
    try:
        arduino = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to Arduino on port {port} at {baud_rate} baud.")
        return arduino
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino: {e}")
        return None

def load_previous_calibration(file_path):
    """Load previous calibration data from a CSV file into a pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Starting with new calibration data.")
        return None

    try:
        df = pd.read_csv(file_path)
        print("Previous calibration data loaded successfully.")
        #print("Columns in the loaded DataFrame:", df.columns)  # Debugging: Print column names
        #print("First few rows of the DataFrame:\n", df.head())  # Debugging: Print first few rows
        return df
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def save_calibration_data(file_path, calibration_data, batch_id):
    """Save calibration data from a pandas DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        # Add the batch_id column to the calibration data
        calibration_data['batch_id'] = batch_id

        # Check if the file already exists
        if os.path.exists(file_path):
            # Append to the existing file without writing the header again
            calibration_data.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # Write a new file with the header
            calibration_data.to_csv(file_path, index=False)
        print(f"Calibration data saved successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def get_solenoid_dt_values(ndt, previous_dt_values=None):
    """Retrieve solenoid opening times (dt values) from the user."""
    if previous_dt_values is not None:
        print("Using previous solenoid dt values:", previous_dt_values)
        return previous_dt_values
    else:
        dt_values = []
        for i in range(ndt):
            while True:
                try:
                    dt = int(input(f"Enter solenoid opening time (ms) dt({i + 1}): "))
                    dt_values.append(dt)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        print("New solenoid dt values entered:", dt_values)
        return dt_values

def collect_water_volumes(arduino, dt_values, num_reps, initial_volume):
    """Collect water volumes for each solenoid opening time."""
    water_volumes = []
    for j, dt in enumerate(dt_values):
        print(f"Testing solenoid opening time: {dt} ms")
        
        # Clear input buffer
        while arduino.in_waiting > 4:
            arduino.readline()
        
        for _ in range(num_reps):
            signal = int(f"2{dt}")
            print(signal)
            arduino.write(str(signal).encode() + b"\n")
            time.sleep(0.5)
        
        while True:
            try:
                cumulative_volume = float(input(f"Enter cumulative water amount (mL) for dt({j + 1}): "))
                water_volumes.append(cumulative_volume - initial_volume)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    return water_volumes

def main():
    print("Welcome to Solenoid Calibration!")
    
    # Configuration
    arduino_port = "COM7"  # Replace with your Arduino's port
    baud_rate = 115200
    r2_threshold = 0.9
    ndt = 4  # Number of solenoid opening times to test
    num_reps = 30
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "solenoid_calibration_results.csv"
    file_path = os.path.join(script_dir, file_name)
    
    # Connect to Arduino
    arduino = connect_to_arduino(arduino_port, baud_rate)
    if not arduino:
        return
    
    # Load previous calibration data
    open_previous = input("Open previous calibration results file? ([0]/[1]): ").strip()
    previous_data = load_previous_calibration(file_path) if open_previous == "1" else None
    
    # Determine the next batch_id
    if previous_data is not None:
        batch_id = previous_data['batch_id'].max() + 1
    else:
        batch_id = 1
    
    # Get solenoid dt values
    use_previous = input("Use previous solenoid dt values? ([0]/[1]): ").strip()
    dt_values = get_solenoid_dt_values(ndt, previous_data['dt'].tolist() if previous_data is not None and use_previous == "1" else None)
    
    # Get initial water volume
    while True:
        try:
            initial_volume = float(input("Enter the starting volume of water (in mL): "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
    
    # Collect water volumes
    water_volumes = collect_water_volumes(arduino, dt_values, num_reps, initial_volume)
    print("Collected water volumes:", water_volumes)
    
    # Perform calibration calculations
    dw_values = np.array(water_volumes) / num_reps * 1000  # Convert to μL
    calibration_fit = np.polyfit(dt_values, dw_values, 1)  # Linear fit: dw = b0 * dt + b1
    
    # Create a pandas DataFrame for calibration data
    calibration_data = pd.DataFrame({
        'dt': dt_values,
        'w': water_volumes,
        'dw': dw_values,
        'b0': [calibration_fit[0]] * len(dt_values),  # Slope
        'b1': [calibration_fit[1]] * len(dt_values)   # Intercept
    })
    
    # Plot results
    plt.figure()
    plt.plot(dt_values, dw_values, '*', label='Data')
    x = np.array([min(dt_values), max(dt_values)])  # Use dt_values for the x-axis range
    plt.plot(x, calibration_fit[0] * x + calibration_fit[1], '--', label=f'dw = {calibration_fit[0]:.2f}*dt + {calibration_fit[1]:.2f}')
    
    if previous_data is not None:
        prev_dw = previous_data['dw']
        prev_fit = [previous_data['b0'].iloc[0], previous_data['b1'].iloc[0]]
        plt.plot(prev_fit[0] * prev_dw + prev_fit[1], prev_dw, 'b--', label='Previous fit')
    
    plt.xlabel('Solenoid ON time (ms)')
    plt.ylabel('Water amount (μL)')
    plt.legend(loc='lower right')
    plt.show(block=False)
    
    # Compute and display R²
    r2 = round(np.corrcoef(dw_values, dt_values)[0, 1] ** 2, 2)
    print(f'R² value: {r2}')
    if r2 < r2_threshold:
        print('RECALIBRATE, DO NOT SAVE!!!')
    
    # Save calibration data
    save = input('Save? (0/[1]): ').strip()
    if not save or save == "1":
        save_calibration_data(file_path, calibration_data, batch_id)

if __name__ == "__main__":
    main()