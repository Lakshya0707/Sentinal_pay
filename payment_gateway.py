
from flask import Flask, render_template_string, request, jsonify
import requests

app = Flask(__name__)
SENTINEL_API = "http://localhost:5000/api/fraud-check"

@app.route('/')
def payment_page():
    return render_template_string(open('templates/gateway.html', encoding='utf-8').read())

@app.route('/process-payment', methods=['POST'])
def process_payment():
    try:
        data = request.json
        resp = requests.post(SENTINEL_API, json=data, timeout=5)
        return jsonify(resp.json()) if resp.status_code == 200 else jsonify({'success': False, 'error': 'API error'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import webbrowser, threading
    print("Payment Gateway at http://localhost:5001")
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:5001")).start()
    app.run(debug=True, port=5001)