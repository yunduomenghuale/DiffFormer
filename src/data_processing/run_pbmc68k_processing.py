import sys
from pathlib import Path

# Add the 'src' directory to the Python path to ensure imports work
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Now, import the refactored functions
from src.data_processing.d02_preprocess_data import preprocess_data
from src.data_processing.d03_annotate_and_simulate import annotate_and_simulate

def main():
    """
    Runs the full data processing pipeline for the pbmc68k dataset.
    """
    dataset_name = "pbmc68k"
    
    print(f"--- Starting full data processing pipeline for '{dataset_name}' ---")
    
    print("\n[Step 1/3] Preprocessing raw data...")
    preprocess_data(dataset_name)
    print("[Step 1/3] Preprocessing complete.")

    print("\n[Step 2/3] Generating training set...")
    annotate_and_simulate(dataset_name, mode='train')
    print("[Step 2/3] Training set generation complete.")

    print("\n[Step 3/3] Generating test set...")
    annotate_and_simulate(dataset_name, mode='test')
    print("[Step 3/3] Test set generation complete.")
    
    print(f"\n--- Pipeline for '{dataset_name}' finished successfully! ---")

if __name__ == "__main__":
    main() 