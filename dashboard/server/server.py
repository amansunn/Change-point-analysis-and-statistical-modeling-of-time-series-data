from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Windows path handling
def load_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw'))
    prices_path = os.path.join(base_dir, 'BrentOilPrices.csv')
    events_path = os.path.join(base_dir, 'events.csv')
    
    prices = pd.read_csv(prices_path, parse_dates=['Date'])
    events = pd.read_csv(events_path, parse_dates=['Date'])
    return prices, events

prices_df, events_df = load_data()

@app.route('/api/prices')
def get_prices():
    return jsonify(prices_df.to_dict(orient='records'))

@app.route('/api/events')
def get_events():
    return jsonify(events_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)