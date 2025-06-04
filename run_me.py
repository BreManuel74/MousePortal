import os
import subprocess
import sys
import csv
from datetime import datetime

def list_files(folder, ext=".py"):
    return [f for f in os.listdir(folder) if f.endswith(ext)]

def list_dirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def select_file(folder_name, ext=".py"):
    folder_path = os.path.join(os.getcwd(), folder_name)
    files = list_files(folder_path, ext)
    print(f"Files in {folder_name}:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}: {file}")
    choice = int(input(f"Select a file from {folder_name} (1-{len(files)}): ")) - 1
    return os.path.join(folder_path, files[choice])

def select_dir(base_folder):
    while True:
        dirs = list_dirs(base_folder)
        print(f"\nCurrent directory: {base_folder}")
        if not dirs:
            #print(f"No directories found in {base_folder}.")
            return base_folder
        print("0: Select this directory")
        for idx, d in enumerate(dirs):
            print(f"{idx + 1}: {d}")
        choice = int(input(f"Select a directory (1-{len(dirs)}), or 0 to use this directory: "))
        if choice == 0:
            return base_folder
        base_folder = os.path.join(base_folder, dirs[choice - 1])

def make_relative_forward_slash(path):
    rel_path = os.path.relpath(path, os.getcwd())
    return rel_path.replace("\\", "/")

def log_run(animal_name, level_file, phase_file, batch_id):
    log_dir = "Progress_Reports"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
    log_file = os.path.join(log_dir, f"{animal_name}_log.csv")
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    # Only log the file names, not full paths
    level_file_name = os.path.basename(level_file)
    phase_file_name = os.path.basename(phase_file)
    # Write header if file does not exist
    write_header = not os.path.exists(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Date", "Time", "Level File", "Phase File", "Batch ID"])
        writer.writerow([date_str, time_str, level_file_name, phase_file_name, batch_id])

def run_phase_with_level(phase_file_path, level_file_path, output_dir=None, batch_id=None, teensy_port=None):
    level_file_path = make_relative_forward_slash(level_file_path)
    print(f"About to run: {phase_file_path} with config: {level_file_path}")
    env = os.environ.copy()
    env["LEVEL_CONFIG_PATH"] = level_file_path
    if output_dir:
        env["OUTPUT_DIR"] = output_dir
    if batch_id is not None:
        env["BATCH_ID"] = str(batch_id)
    if teensy_port:
        env["TEENSY_PORT"] = teensy_port
    subprocess.run([sys.executable, phase_file_path], env=env)
    #print("Phase file loaded!")

if __name__ == "__main__":
    animal_name = input("Enter animal name: ")
    level_file = select_file('Levels', '.json')
    phase_file = select_file('Phases', '.py')
    output_dir = select_dir(os.getcwd())
    batch_id = input("Enter batch ID number: ")
    teensy_port = input("Enter Teensy port (e.g., COM3): ")
    #print(f"Running {phase_file} with level config {level_file}, OUTPUT_DIR {output_dir}, and BATCH_ID {batch_id}...")
    log_run(animal_name, level_file, phase_file, batch_id)
    run_phase_with_level(phase_file, level_file, output_dir, batch_id, teensy_port)
