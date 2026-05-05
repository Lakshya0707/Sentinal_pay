from model import load_dataset, FraudPreprocessor, train_model, save_artifacts

if __name__ == "__main__":
    print("="*60)
    print("  SENTINEL AI — Fraud Detection Model Training")
    print("="*60)

    print("\n🔧 Step 1: Loading dataset from Excel...")
    df = load_dataset("fraud_dataset.xlsx")

    print("\n🔧 Step 2: Training Voting Ensemble (GBM + RF + Logistic)...")
    print("   Applying oversampling for class balancing...")
    preprocessor = FraudPreprocessor()
    model, metrics, _ = train_model(df, preprocessor)

    print("\n🔧 Step 3: Saving model artifacts...")
    save_artifacts(model, preprocessor, metrics)

    print("\n" + "="*60)
    print("✅ Training complete! Now run:  python app.py")
    print("   Then open:  http://localhost:5000")
    print("="*60)
