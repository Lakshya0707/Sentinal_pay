=====================================
  SENTINEL PAY — Fraud Detection
=====================================

SETUP:
  pip install -r requirements.txt

HOW TO RUN:
  Step 1: Train model (only once)
    python train.py

  Step 2: Start main dashboard
    python app.py
    Open: http://localhost:5000

  Step 3: Start payment gateway (optional, separate terminal)
    python payment_gateway.py
    Open: http://localhost:5001

ADMIN PANEL:
  URL:      http://localhost:5000/admin/login
  Username: admin
  Password: admin123

FRAUD THRESHOLDS:
  Score < 50   → APPROVED
  Score 50-70  → MIGHT BE FRAUD (review)
  Score > 70   → BLOCKED

FEATURES USED (16):
  payment_type, merchant_category, amount, hour, day_of_week,
  velocity, mins_since_last_txn, failed_attempts,
  is_vpn, is_new_device, is_intl_txn, is_upi_new_payee,
  is_first_time_payee, is_odd_hour, user_avg_amount, amount_vs_avg_ratio
=====================================
