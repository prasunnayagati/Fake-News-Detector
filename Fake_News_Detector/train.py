from src.training import AetherNeuralEngine
from src.data_generator import create_sample_data
import os
import time

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_console()
    print("="*60)
    print("       AETHER NEURAL SENTINEL - MISSION CONTROL v2.5.0")
    print("="*60)
    
    if not os.path.exists('data/raw/True.csv') or not os.path.exists('data/raw/Fake.csv'):
        print("[!] SIGNAL LOST: Raw archives not found.")
        print("[!] INITIATING SAMPLE DATA GENERATION...")
        create_sample_data()
        time.sleep(1)
    
    print("[+] INITIALIZING NEURAL ENGINE...")
    engine = AetherNeuralEngine()
    
    print("[+] SYNCHRONIZING WITH ARCHIVES...")
    df = engine.prepare_data('data/raw/True.csv', 'data/raw/Fake.csv')
    
    print("[+] CALIBRATING NEURAL WEIGHTS...")
    metrics = engine.train(df)
    
    print("\n" + "-"*60)
    print("     NEURAL CALIBRATION COMPLETE - PERFORMANCE REPORT")
    print("-"*60)
    print(f" ACCURACY SCORE: {metrics['accuracy']:.4%}")
    print(f" ROC-AUC METRIC: {metrics['roc_auc']:.4%}")
    print("\n[CLASSIFICATION_LOGS]")
    print(metrics['report'])
    print("="*60)
    print("   SENTINEL READY - DEPLOY WITH 'streamlit run app.py'")
    print("="*60)

if __name__ == "__main__":
    main()
