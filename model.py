import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_score, recall_score, f1_score)
from sklearn.utils import resample

np.random.seed(42)

PAYMENT_TYPES = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"]
MERCHANT_CATS = ["Groceries", "Electronics", "Travel", "Food & Dining", "Clothing",
                 "Healthcare", "Entertainment", "Education", "Fuel", "Jewelry",
                 "Crypto Exchange", "Casino"]
HIGH_RISK_CATS = {"Crypto Exchange", "Casino", "Jewelry", "Electronics", "Travel"}

PAYMENT_AVG_AMOUNTS = {
    "UPI":          ("small",   50,   30000),
    "Credit Card":  ("medium", 200,   80000),
    "Debit Card":   ("medium", 100,   30000),
    "Net Banking":  ("large",  500,  200000),
    "Wallet":       ("small",   20,    2000),
}

CUSTOMERS = None

def load_customers(excel_path="fraud_dataset.xlsx"):
    global CUSTOMERS
    df = pd.read_excel(excel_path)
    CUSTOMERS = df.groupby('customer_id').agg(
        customer_name=('customer_name', 'first'),
        bank=('bank', 'first'),
        card_number=('card_number', 'first'),
        upi_id=('upi_id', 'first'),
        user_avg_amount=('user_avg_amount', 'first'),
        preferred_payment=('payment_type', lambda x: x.mode()[0]),
    ).reset_index().to_dict('records')
    return CUSTOMERS

FEATURE_COLS = [
    "payment_type_enc", "merchant_cat_enc",
    "amount", "hour", "day_of_week",
    "velocity", "mins_since_last_txn", "failed_attempts",
    "is_vpn", "is_new_device", "is_intl_txn",
    "is_upi_new_payee", "is_first_time_payee",
    "is_odd_hour", "user_avg_amount", "amount_vs_avg_ratio"
]



class EnsembleModel:
    """Soft ensemble for realistic probability spread."""
    def __init__(self, gbm, rf, lr):
        self.gbm = gbm
        self.rf  = rf
        self.lr  = lr

    def predict_proba(self, X):
        import numpy as np
        p = (3*self.gbm.predict_proba(X)[:,1] +
             2*self.rf.predict_proba(X)[:,1] +
             1*self.lr.predict_proba(X)[:,1]) / 6
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

class FraudPreprocessor:
    def __init__(self):
        self.le_payment  = LabelEncoder()
        self.le_merchant = LabelEncoder()
        self.scaler      = StandardScaler()
        self.feature_names = None
        self._fitted     = False

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        df["payment_type_enc"]  = self.le_payment.fit_transform(df["payment_type"])
        df["merchant_cat_enc"]  = self.le_merchant.fit_transform(df["merchant_category"])
        self.feature_names      = FEATURE_COLS
        X = df[FEATURE_COLS].values
        X = self.scaler.fit_transform(X)
        self._fitted = True
        return X, df["is_fraud"].values

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        for val in df["payment_type"].unique():
            if val not in self.le_payment.classes_:
                df["payment_type"] = df["payment_type"].replace(val, self.le_payment.classes_[0])
        for val in df["merchant_category"].unique():
            if val not in self.le_merchant.classes_:
                df["merchant_category"] = df["merchant_category"].replace(val, self.le_merchant.classes_[0])
        df["payment_type_enc"]  = self.le_payment.transform(df["payment_type"])
        df["merchant_cat_enc"]  = self.le_merchant.transform(df["merchant_category"])
        X = df[FEATURE_COLS].values
        return self.scaler.transform(X)


def load_dataset(excel_path="fraud_dataset.xlsx"):
    df = pd.read_excel(excel_path)
    print(f"   Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"   Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
    return df


def train_model(df: pd.DataFrame, preprocessor: FraudPreprocessor):
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Manual oversampling (replaces SMOTE)
    X_tr_df = pd.DataFrame(X_train)
    y_tr_s  = pd.Series(y_train)
    X_maj   = X_tr_df[y_tr_s == 0]
    X_min   = X_tr_df[y_tr_s == 1]
    X_min_up = resample(X_min, replace=True, n_samples=len(X_maj), random_state=42)
    X_res   = np.vstack([X_maj.values, X_min_up.values])
    y_res   = np.array([0]*len(X_maj) + [1]*len(X_min_up))

    gbm = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42)
    rf  = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        random_state=42, n_jobs=-1)
    lr  = LogisticRegression(C=0.5, max_iter=500, random_state=42)

    model = VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf), ("lr", lr)],
        voting="soft", weights=[3, 2, 1])
    model.fit(X_res, y_res)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(float(np.mean(y_pred == y_test)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall":    round(float(recall_score(y_test, y_pred)), 4),
        "f1":        round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
        "report":    classification_report(y_test, y_pred,
                        target_names=["Legitimate", "Fraud"]),
    }
    print(f"\n MODEL TRAINING COMPLETE")
    print(f"   Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall   : {metrics['recall']*100:.2f}%")
    print(f"   F1 Score : {metrics['f1']*100:.2f}%")
    print(f"   ROC-AUC  : {metrics['roc_auc']*100:.2f}%\n")
    return model, metrics, (X_test, y_test, y_pred, y_proba)


def save_artifacts(model, preprocessor, metrics, path="models"):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/fraud_model.pkl",   "wb") as f: pickle.dump(model, f)
    with open(f"{path}/preprocessor.pkl",  "wb") as f: pickle.dump(preprocessor, f)
    with open(f"{path}/metrics.pkl",       "wb") as f: pickle.dump(metrics, f)
    print(f"Artifacts saved to ./{path}/")


def load_artifacts(path="models"):
    with open(f"{path}/fraud_model.pkl",  "rb") as f: model        = pickle.load(f)
    with open(f"{path}/preprocessor.pkl", "rb") as f: preprocessor = pickle.load(f)
    with open(f"{path}/metrics.pkl",      "rb") as f: metrics      = pickle.load(f)
    return model, preprocessor, metrics


def predict_transaction(transaction: dict, model, preprocessor) -> dict:
    df    = pd.DataFrame([transaction])
    X     = preprocessor.transform(df)
    proba = float(model.predict_proba(X)[0][1])
    raw   = proba * 100

    # Score stretching - maps model's raw probabilities to display score
    # Raw distribution: legit=0-15%, borderline=30-85%, fraud=85-100%
    # Maps raw 0-15  → score 0-40   (APPROVED)
    # Maps raw 15-50 → score 40-55  (APPROVED/border)
    # Maps raw 50-82 → score 55-70  (MIGHT_BE_FRAUD zone)
    # Maps raw 82+   → score 70-100 (BLOCKED)
    if raw <= 15:
        score = raw * (40.0 / 15.0)
    elif raw <= 50:
        score = 40.0 + (raw - 15.0) * (15.0 / 35.0)
    elif raw <= 82:
        score = 55.0 + (raw - 50.0) * (15.0 / 32.0)
    else:
        score = 70.0 + (raw - 82.0) * (30.0 / 18.0)

    score = round(min(max(score, 0), 100), 1)

    if score >= 70:
        risk = "CRITICAL"; blocked = True;  status = "BLOCKED"
    elif score >= 50:
        risk = "HIGH";     blocked = False; status = "MIGHT_BE_FRAUD"
    elif score >= 35:
        risk = "MEDIUM";   blocked = False; status = "REVIEW"
    else:
        risk = "LOW";      blocked = False; status = "APPROVED"

    return {
        "fraud_probability": round(proba, 4),
        "fraud_score":       score,
        "is_fraud":          score >= 50,
        "risk_level":        risk,
        "status":            status,
        "blocked":           blocked,
    }


if __name__ == "__main__":
    print("="*60)
    print("  SENTINEL AI - Training on Excel Dataset")
    print("="*60)
    df           = load_dataset("fraud_dataset.xlsx")
    preprocessor = FraudPreprocessor()
    model, metrics, _ = train_model(df, preprocessor)
    save_artifacts(model, preprocessor, metrics)
