import os
import subprocess
import sys

def list_files(folder, ext=".py"):
    return [f for f in os.listdir(folder) if f.endswith(ext)]

def select_file(folder_name, ext=".py"):
    folder_path = os.path.join(os.getcwd(), folder_name)
    files = list_files(folder_path, ext)
    print(f"Files in {folder_name}:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}: {file}")
    choice = int(input(f"Select a file from {folder_name} (1-{len(files)}): ")) - 1
    return os.path.join(folder_path, files[choice])

def make_relative_forward_slash(path):
    rel_path = os.path.relpath(path, os.getcwd())
    return rel_path.replace("\\", "/")

def run_phase_with_level(phase_file_path, level_file_path):
    level_file_path = make_relative_forward_slash(level_file_path)
    print(f"About to run: {phase_file_path} with config: {level_file_path}")
    env = os.environ.copy()
    env["LEVEL_CONFIG_PATH"] = level_file_path
    subprocess.run([sys.executable, phase_file_path], env=env)
    print("Phase file loaded!")

if __name__ == "__main__":
    level_file = select_file('Levels', '.json')
    phase_file = select_file('Phases', '.py')
    print(f"Running {phase_file} with level config {level_file}...")
    run_phase_with_level(phase_file, level_file)
