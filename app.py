
import os, json, time, random, threading, pickle, webbrowser
from datetime import datetime
from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
from functools import wraps
import numpy as np
import pandas as pd

# ── Bootstrap model ──────────────────────────────────────────────────────
MODEL_PATH = "models/fraud_model.pkl"
EXCEL_PATH = "fraud_dataset.xlsx"

if not os.path.exists(MODEL_PATH):
    print("  No saved model found. Training now (~60 sec)...")
    from model import load_dataset, FraudPreprocessor, train_model, save_artifacts
    df           = load_dataset(EXCEL_PATH)
    preprocessor = FraudPreprocessor()
    model_obj, metrics_obj, _ = train_model(df, preprocessor)
    save_artifacts(model_obj, preprocessor, metrics_obj)

from model import (load_artifacts, load_customers, predict_transaction,
                   PAYMENT_TYPES, MERCHANT_CATS, HIGH_RISK_CATS, PAYMENT_AVG_AMOUNTS)

model, preprocessor, metrics = load_artifacts()
CUSTOMERS = load_customers(EXCEL_PATH)

# ── Flask ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "sentinel_fraud_admin_2024"

# ── State ─────────────────────────────────────────────────────────────────
transaction_log = []
stats = {"total": 0, "fraud": 0, "blocked": 0, "maybe_fraud": 0, "saved_amount": 0.0}
sim_running = True
TX_COUNTER  = 1000
LOCK        = threading.Lock()

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def gen_tx_id():
    global TX_COUNTER
    with LOCK:
        TX_COUNTER += 1
        return f"TXN{TX_COUNTER:07d}"


def generate_live_transaction(force_fraud=False, customer=None):
    if customer is None:
        customer = random.choice(CUSTOMERS)

    ptype    = customer["preferred_payment"] if random.random() < 0.7 else random.choice(PAYMENT_TYPES)
    _, lo, hi = PAYMENT_AVG_AMOUNTS.get(ptype, ("medium", 100, 50000))
    user_avg = float(customer["user_avg_amount"])
    user_std = user_avg * 0.4

    rand_val = random.random()
    is_fraud = force_fraud or (rand_val < 0.05)
    is_borderline = (not is_fraud) and (rand_val < 0.20)  # 15% extra borderline cases

    if is_fraud:
        amount        = random.choice([
            random.uniform(user_avg * 3, user_avg * 12),
            random.uniform(1, 50)
        ])
        hour          = random.choice([0,1,2,3,4,5,22,23]) if random.random() < 0.6 else random.randint(0,23)
        velocity      = random.randint(6, 18)
        mins_gap      = round(random.uniform(0.5, 10), 2)
        failed        = random.choices([0,1,2,3,4], weights=[20,25,25,18,12])[0]
        is_vpn        = 1 if random.random() < 0.65 else 0
        is_new_device = 1 if random.random() < 0.60 else 0
        is_intl       = 1 if random.random() < 0.45 else 0
        merchant      = random.choice(list(HIGH_RISK_CATS) + ["Electronics","Jewelry"])
        is_upi_new    = 1 if random.random() < 0.75 else 0
        is_first      = 1 if random.random() < 0.80 else 0
    else:
        amount        = max(10, np.random.normal(user_avg, user_std))
        hour          = int(np.clip(np.random.normal(14, 4), 6, 22))
        velocity      = random.choices([1,2,3,4,5], weights=[50,25,15,7,3])[0]
        mins_gap      = round(random.uniform(30, 400), 2)
        failed        = random.choices([0,1,2,3], weights=[75,15,7,3])[0]
        is_vpn        = 1 if random.random() < 0.03 else 0
        is_new_device = 1 if random.random() < 0.05 else 0
        is_intl       = 1 if random.random() < 0.04 else 0
        merchant      = random.choice([c for c in MERCHANT_CATS if c not in HIGH_RISK_CATS])
        is_upi_new    = 1 if random.random() < 0.08 else 0
        is_first      = 1 if random.random() < 0.10 else 0

    if is_borderline:
        # Borderline: some suspicious flags but not clearly fraud
        amount        = random.uniform(user_avg * 1.5, user_avg * 3.5)
        hour          = random.choice([6,7,21,22,23,0]) if random.random() < 0.5 else random.randint(8,20)
        velocity      = random.randint(4, 7)
        mins_gap      = round(random.uniform(8, 35), 2)
        failed        = random.choices([0,1,2], weights=[40,40,20])[0]
        is_vpn        = 1 if random.random() < 0.40 else 0
        is_new_device = 1 if random.random() < 0.35 else 0
        is_intl       = 1 if random.random() < 0.20 else 0
        merchant      = random.choice(["Electronics","Jewelry","Travel","Clothing","Food & Dining"])
        is_upi_new    = 1 if random.random() < 0.35 else 0
        is_first      = 1 if random.random() < 0.30 else 0

    amount        = round(float(amount), 2)
    dow           = random.randint(0, 6)
    is_odd_hour   = 1 if hour < 6 or hour > 22 else 0
    amount_ratio  = round(amount / max(user_avg, 1), 3)

    tx_dict = {
        "payment_type":       ptype,
        "merchant_category":  merchant,
        "amount":             amount,
        "hour":               hour,
        "day_of_week":        dow,
        "velocity":           velocity,
        "mins_since_last_txn": mins_gap,
        "failed_attempts":    failed,
        "is_vpn":             is_vpn,
        "is_new_device":      is_new_device,
        "is_intl_txn":        is_intl,
        "is_upi_new_payee":   is_upi_new,
        "is_first_time_payee": is_first,
        "is_odd_hour":        is_odd_hour,
        "user_avg_amount":    round(user_avg, 2),
        "amount_vs_avg_ratio": amount_ratio,
    }

    result = predict_transaction(tx_dict, model, preprocessor)

    card_num = customer["card_number"] if ptype in ("Credit Card","Debit Card") else ""
    upi_id   = customer["upi_id"]      if ptype == "UPI" else ""

    now = datetime.now()
    record = {
        "id":               gen_tx_id(),
        "timestamp":        now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_ms":     int(now.timestamp() * 1000),
        "cardholder":       customer["customer_name"],
        "customer_id":      customer["customer_id"],
        "bank":             customer["bank"],
        "card_number":      card_num,
        "upi_id":           upi_id,
        "payment_type":     ptype,
        "amount":           amount,
        "merchant_category": merchant,
        "hour":             hour,
        "velocity":         velocity,
        "mins_since_last_txn": mins_gap,
        "failed_attempts":  failed,
        "is_new_device":    is_new_device,
        "is_vpn":           is_vpn,
        "is_intl":          is_intl,
        "is_odd_hour":      is_odd_hour,
        "amount_vs_avg_ratio": amount_ratio,
        "user_avg_amount":  round(user_avg, 2),
        "fraud_score":      result["fraud_score"],
        "fraud_probability": result["fraud_probability"],
        "risk_level":       result["risk_level"],
        "status":           result["status"],
        "is_fraud":         result["is_fraud"],
        "blocked":          result["blocked"],
    }
    return record


# ── Simulation Thread ─────────────────────────────────────────────────────
def simulation_thread():
    while True:
        if sim_running:
            tx = generate_live_transaction()
            with LOCK:
                transaction_log.insert(0, tx)
                if len(transaction_log) > 1000:
                    transaction_log.pop()
                stats["total"] += 1
                if tx["is_fraud"]:
                    stats["fraud"] += 1
                    if tx["status"] == "MIGHT_BE_FRAUD":
                        stats["maybe_fraud"] += 1
                if tx["blocked"]:
                    stats["blocked"]      += 1
                    stats["saved_amount"] += tx["amount"]
        time.sleep(1.5)


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index_polling.html", metrics=metrics)


@app.route("/api/transactions")
def api_transactions():
    limit       = int(request.args.get("limit", 100))
    name_filter = request.args.get("name", "").strip().lower()
    start_time  = request.args.get("start_time", "").strip()
    end_time    = request.args.get("end_time", "").strip()

    with LOCK:
        txs = list(transaction_log[:500])

    if name_filter:
        txs = [t for t in txs if name_filter in t["cardholder"].lower()]

    if start_time and end_time:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            st = datetime.strptime(f"{today} {start_time}", "%Y-%m-%d %H:%M:%S")
            et = datetime.strptime(f"{today} {end_time}",   "%Y-%m-%d %H:%M:%S")
            txs = [t for t in txs
                   if st <= datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S") <= et]
        except:
            pass

    return jsonify(txs[:limit])


@app.route("/api/customers")
def api_customers():
    names = sorted(set(c["customer_name"] for c in CUSTOMERS))
    return jsonify(names)


@app.route("/api/stats")
def api_stats():
    with LOCK:
        det_rate = round(stats["fraud"] / max(stats["total"], 1) * 100, 1)
        return jsonify({
            **stats,
            "detection_rate":  det_rate,
            "model_accuracy":  round(metrics["accuracy"] * 100, 2),
            "model_auc":       round(metrics["roc_auc"] * 100, 2),
        })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    try:
        result = predict_transaction(data, model, preprocessor)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate/toggle", methods=["POST"])
def toggle_simulation():
    global sim_running
    sim_running = not sim_running
    return jsonify({"status": "running" if sim_running else "paused"})


@app.route("/api/simulate/fraud", methods=["POST"])
def inject_fraud():
    tx = generate_live_transaction(force_fraud=True)
    with LOCK:
        transaction_log.insert(0, tx)
        stats["total"] += 1
        stats["fraud"] += 1
        if tx["status"] == "MIGHT_BE_FRAUD":
            stats["maybe_fraud"] += 1
        if tx["blocked"]:
            stats["blocked"]      += 1
            stats["saved_amount"] += tx["amount"]
    return jsonify(tx)


@app.route("/api/fraud-check", methods=["POST"])
def api_fraud_check():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data"}), 400

        ptype         = data.get("payment_type", "UPI")
        amount        = float(data.get("amount", 0))
        merchant      = data.get("merchant_category", "Groceries")
        customer_name = data.get("customer_name", "Unknown")
        hour          = int(data.get("hour", datetime.now().hour))
        dow           = datetime.now().weekday()
        is_odd_hour   = 1 if hour < 6 or hour > 22 else 0

        # Find customer avg if known
        cust_match = next((c for c in CUSTOMERS if c["customer_name"] == customer_name), None)
        user_avg   = float(cust_match["user_avg_amount"]) if cust_match else 5000.0
        amount_ratio = round(amount / max(user_avg, 1), 3)

        card_num = data.get("card_number", cust_match["card_number"] if cust_match else "")
        upi_id   = data.get("upi_id",   cust_match["upi_id"]   if cust_match else "")
        bank     = cust_match["bank"] if cust_match else "Unknown"

        tx_dict = {
            "payment_type":       ptype,
            "merchant_category":  merchant,
            "amount":             amount,
            "hour":               hour,
            "day_of_week":        dow,
            "velocity":           int(data.get("velocity", 1)),
            "mins_since_last_txn": float(data.get("mins_since_last_txn", 120)),
            "failed_attempts":    int(data.get("failed_attempts", 0)),
            "is_vpn":             int(data.get("is_vpn", 0)),
            "is_new_device":      int(data.get("is_new_device", 0)),
            "is_intl_txn":        int(data.get("is_intl_txn", 0)),
            "is_upi_new_payee":   int(data.get("is_upi_new_payee", 0)),
            "is_first_time_payee": int(data.get("is_first_time_payee", 0)),
            "is_odd_hour":        is_odd_hour,
            "user_avg_amount":    round(user_avg, 2),
            "amount_vs_avg_ratio": amount_ratio,
        }

        result = predict_transaction(tx_dict, model, preprocessor)

        now = datetime.now()
        record = {
            "id":               gen_tx_id(),
            "timestamp":        now.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_ms":     int(now.timestamp() * 1000),
            "cardholder":       customer_name,
            "customer_id":      cust_match["customer_id"] if cust_match else "MANUAL",
            "bank":             bank,
            "card_number":      card_num if ptype in ("Credit Card","Debit Card") else "",
            "upi_id":           upi_id if ptype == "UPI" else "",
            "payment_type":     ptype,
            "amount":           amount,
            "merchant_category": merchant,
            "hour":             hour,
            "velocity":         tx_dict["velocity"],
            "mins_since_last_txn": tx_dict["mins_since_last_txn"],
            "failed_attempts":  tx_dict["failed_attempts"],
            "is_new_device":    tx_dict["is_new_device"],
            "is_vpn":           tx_dict["is_vpn"],
            "is_intl":          tx_dict["is_intl_txn"],
            "is_odd_hour":      is_odd_hour,
            "amount_vs_avg_ratio": amount_ratio,
            "user_avg_amount":  round(user_avg, 2),
            "fraud_score":      result["fraud_score"],
            "fraud_probability": result["fraud_probability"],
            "risk_level":       result["risk_level"],
            "status":           result["status"],
            "is_fraud":         result["is_fraud"],
            "blocked":          result["blocked"],
        }

        with LOCK:
            transaction_log.insert(0, record)
            if len(transaction_log) > 1000:
                transaction_log.pop()
            stats["total"] += 1
            if record["is_fraud"]:
                stats["fraud"] += 1
                if record["status"] == "MIGHT_BE_FRAUD":
                    stats["maybe_fraud"] += 1
            if record["blocked"]:
                stats["blocked"]      += 1
                stats["saved_amount"] += record["amount"]

        return jsonify({
            "success":        True,
            "transaction_id": record["id"],
            "fraud_result": {
                "fraud_score":      result["fraud_score"],
                "fraud_probability": result["fraud_probability"],
                "risk_level":       result["risk_level"],
                "status":           result["status"],
                "is_fraud":         result["is_fraud"],
                "blocked":          result["blocked"],
            },
            "timestamp": record["timestamp"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/status")
def api_status():
    return jsonify({
        "status":       "online",
        "model_loaded": model is not None,
        "total":        stats.get("total", 0),
        "fraud":        stats.get("fraud", 0),
    })


# ── Admin ─────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "logged_in" not in session:
            flash("Please login first")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if (request.form.get("username") == ADMIN_USERNAME and
                request.form.get("password") == ADMIN_PASSWORD):
            session["logged_in"] = True
            session["username"]  = request.form.get("username")
            return redirect(url_for("admin_dashboard"))
        flash("Invalid credentials!")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    return render_template("admin_dashboard.html", stats=stats, metrics=metrics, session=session)


@app.route("/admin/retrain", methods=["POST"])
@login_required
def admin_retrain():
    try:
        from model import load_dataset, FraudPreprocessor, train_model, save_artifacts
        df           = load_dataset(EXCEL_PATH)
        preprocessor = FraudPreprocessor()
        new_model, new_metrics, _ = train_model(df, preprocessor)
        save_artifacts(new_model, preprocessor, new_metrics)
        global model, metrics
        model   = new_model
        metrics = new_metrics
        return jsonify({"success": True, "message": "Model retrained!", "metrics": new_metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ── Entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim_thread = threading.Thread(target=simulation_thread, daemon=True)
    sim_thread.start()
    print("\n" + "="*60)
    print("  Sentinel Pay running at http://localhost:5000")
    print("  Admin Panel:  http://localhost:5000/admin/login")
    print("  Username: admin | Password: admin123")
    print("="*60 + "\n")
    def open_browser():
        webbrowser.open_new("http://localhost:5000")
    threading.Timer(1, open_browser).start()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
