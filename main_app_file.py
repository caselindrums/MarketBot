#!/usr/bin/env python3
"""
Advanced Order Book Analysis Bot for Binance.US
Optimized for Render deployment with simplified architecture
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import threading
from collections import deque, defaultdict
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# HTML template embedded for Render deployment
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Order Book Analysis | Binance.US</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .header h1 {
            color: #2d3748;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            padding: 10px 15px;
            border-radius: 8px;
            font-weight: 600;
        }
        .status {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
        }
        .status.connected {
            background: rgba(72, 187, 120, 0.2);
            color: #2f855a;
        }
        .status.disconnected {
            background: rgba(245, 101, 101, 0.2);
            color: #c53030;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .chart-container {
            height: 300px;
            margin: 20px 0;
        }
        .symbol-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .symbol-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.3s ease;
            cursor: pointer;
            border-left: 4px solid #a0aec0;
        }
        .symbol-card:hover {
            transform: translateY(-3px);
        }
        .symbol-card.bullish {
            border-left-color: #48bb78;
        }
        .symbol-card.bearish {
            border-left-color: #f56565;
        }
        .symbol-card.selected {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }
        .symbol-name {
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .price {
            font-size: 1.2em;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 10px;
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .metric {
            text-align: center;
        }
        .metric-label {
            font-size: 0.8em;
            color: #718096;
            margin-bottom: 4px;
        }
        .metric-value {
            font-weight: 600;
        }
        .signal {
            text-align: center;
            margin-top: 12px;
            padding: 8px;
            border-radius: 8px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .signal.buy {
            background: rgba(72, 187, 120, 0.2);
            color: #2f855a;
        }
        .signal.sell {
            background: rgba(245, 101, 101, 0.2);
            color: #c53030;
        }
        .signal.hold {
            background: rgba(160, 174, 192, 0.2);
            color: #4a5568;
        }
        .loading {
            text-align: center;
            padding: 40px;
        }
        .spinner {
            border: 4px solid #e2e8f0;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Advanced Order Book Analysis</h1>
            <p>Real-time market microstructure analysis with ML-based price prediction for Binance.US</p>
            
            <div class="controls">
                <select id="symbolSelect">
                    <option value="">Loading symbols...</option>
                </select>
                <button class="btn" id="refreshBtn">🔄 Refresh Data</button>
                <div class="status" id="connectionStatus">
                    <span>🔴</span> Disconnected
                </div>
            </div>
        </div>

        <div class="dashboard">
            <div class="card">
                <h3>📊 Order Book Analysis</h3>
                <div class="chart-container">
                    <canvas id="orderbookChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h3>📈 Price & Imbalance Trends</h3>
                <div class="chart-container">
                    <canvas id="trendsChart"></canvas>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>💹 Symbol Overview</h3>
            <div id="symbolsLoading" class="loading">
                <div class="spinner"></div>
                <p>Loading market data...</p>
            </div>
            <div class="symbol-grid" id="symbolGrid" style="display: none;"></div>
        </div>
    </div>

    <script>
        class OrderBookAnalyzer {
            constructor() {
                this.socket = io();
                this.selectedSymbol = 'BTCUSDT';
                this.symbolData = new Map();
                this.charts = {};
                
                this.initializeSocketHandlers();
                this.initializeUI();
                this.initializeCharts();
                this.loadSymbols();
            }

            initializeSocketHandlers() {
                this.socket.on('connect', () => {
                    console.log('Connected to server');
                    this.updateConnectionStatus(true);
                });

                this.socket.on('disconnect', () => {
                    console.log('Disconnected from server');
                    this.updateConnectionStatus(false);
                });

                this.socket.on('orderbook_update', (data) => {
                    this.handleOrderBookUpdate(data);
                });
            }

            initializeUI() {
                document.getElementById('symbolSelect').addEventListener('change', (e) => {
                    this.selectedSymbol = e.target.value;
                    this.updateSelectedSymbol();
                });

                document.getElementById('refreshBtn').addEventListener('click', () => {
                    this.refreshData();
                });
            }

            initializeCharts() {
                const orderbookCtx = document.getElementById('orderbookChart').getContext('2d');
                this.charts.orderbook = new Chart(orderbookCtx, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Bids',
                            data: [],
                            backgroundColor: 'rgba(72, 187, 120, 0.8)',
                        }, {
                            label: 'Asks',
                            data: [],
                            backgroundColor: 'rgba(245, 101, 101, 0.8)',
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Order Book Depth'
                            }
                        }
                    }
                });

                const trendsCtx = document.getElementById('trendsChart').getContext('2d');
                this.charts.trends = new Chart(trendsCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Mid Price',
                            data: [],
                            borderColor: 'rgba(102, 126, 234, 1)',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            tension: 0.4
                        }, {
                            label: 'Imbalance',
                            data: [],
                            borderColor: 'rgba(237, 137, 54, 1)',
                            backgroundColor: 'rgba(237, 137, 54, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Price Movement & Order Flow Imbalance'
                            }
                        }
                    }
                });
            }

            async loadSymbols() {
                try {
                    const response = await fetch('/api/symbols');
                    const data = await response.json();
                    
                    const select = document.getElementById('symbolSelect');
                    select.innerHTML = '';
                    
                    data.symbols.forEach(symbol => {
                        const option = document.createElement('option');
                        option.value = symbol;
                        option.textContent = symbol;
                        if (symbol === this.selectedSymbol) {
                            option.selected = true;
                        }
                        select.appendChild(option);
                    });

                    document.getElementById('symbolsLoading').style.display = 'none';
                    document.getElementById('symbolGrid').style.display = 'grid';
                    
                } catch (error) {
                    console.error('Error loading symbols:', error);
                }
            }

            handleOrderBookUpdate(data) {
                const { symbol, data: symbolInfo, orderbook } = data;
                
                if (!this.symbolData.has(symbol)) {
                    this.symbolData.set(symbol, { history: [] });
                }
                
                const symbolData = this.symbolData.get(symbol);
                symbolData.latest = symbolInfo;
                symbolData.orderbook = orderbook;
                
                symbolData.history.push({
                    timestamp: new Date(symbolInfo.timestamp * 1000),
                    price: symbolInfo.mid_price,
                    imbalance: symbolInfo.weighted_imbalance
                });
                
                if (symbolData.history.length > 50) {
                    symbolData.history.shift();
                }

                this.updateSymbolCard(symbol, symbolInfo);
                
                if (symbol === this.selectedSymbol) {
                    this.updateCharts(symbolInfo, orderbook);
                }
            }

            updateSymbolCard(symbol, data) {
                let card = document.querySelector(`[data-symbol="${symbol}"]`);
                
                if (!card) {
                    card = this.createSymbolCard(symbol);
                    document.getElementById('symbolGrid').appendChild(card);
                }

                const isSelected = symbol === this.selectedSymbol;
                const signalClass = this.getSignalClass(data.signal);
                card.className = `symbol-card ${signalClass} ${isSelected ? 'selected' : ''}`;
                
                card.querySelector('.symbol-name').textContent = symbol;
                card.querySelector('.price').textContent = `$${data.mid_price?.toFixed(4) || 'N/A'}`;
                card.querySelector('.imbalance-value').textContent = `${(data.weighted_imbalance * 100)?.toFixed(2) || 'N/A'}%`;
                card.querySelector('.prediction-value').textContent = `${(data.probability * 100)?.toFixed(1) || 'N/A'}%`;
                
                const signalEl = card.querySelector('.signal');
                signalEl.className = `signal ${this.getSignalClass(data.signal)}`;
                signalEl.textContent = data.signal || 'HOLD';
            }

            createSymbolCard(symbol) {
                const card = document.createElement('div');
                card.className = 'symbol-card';
                card.setAttribute('data-symbol', symbol);
                card.addEventListener('click', () => {
                    this.selectedSymbol = symbol;
                    this.updateSelectedSymbol();
                });

                card.innerHTML = `
                    <div class="symbol-name">${symbol}</div>
                    <div class="price">$0.0000</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Imbalance</div>
                            <div class="metric-value imbalance-value">0%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Prediction</div>
                            <div class="metric-value prediction-value">50%</div>
                        </div>
                    </div>
                    <div class="signal hold">HOLD</div>
                `;

                return card;
            }

            updateCharts(data, orderbook) {
                const trendsChart = this.charts.trends;
                const now = new Date(data.timestamp * 1000);
                
                trendsChart.data.labels.push(now.toLocaleTimeString());
                trendsChart.data.datasets[0].data.push(data.mid_price);
                trendsChart.data.datasets[1].data.push(data.weighted_imbalance);
                
                if (trendsChart.data.labels.length > 30) {
                    trendsChart.data.labels.shift();
                    trendsChart.data.datasets[0].data.shift();
                    trendsChart.data.datasets[1].data.shift();
                }
                
                trendsChart.update('none');
            }

            updateSelectedSymbol() {
                document.getElementById('symbolSelect').value = this.selectedSymbol;
                
                document.querySelectorAll('.symbol-card').forEach(card => {
                    card.classList.remove('selected');
                    if (card.getAttribute('data-symbol') === this.selectedSymbol) {
                        card.classList.add('selected');
                    }
                });
            }

            getSignalClass(signal) {
                switch (signal?.toLowerCase()) {
                    case 'buy': return 'bullish';
                    case 'sell': return 'bearish';
                    default: return 'neutral';
                }
            }

            updateConnectionStatus(connected) {
                const status = document.getElementById('connectionStatus');
                if (connected) {
                    status.className = 'status connected';
                    status.innerHTML = '<span>🟢</span> Connected';
                } else {
                    status.className = 'status disconnected';
                    status.innerHTML = '<span>🔴</span> Disconnected';
                }
            }

            async refreshData() {
                try {
                    const response = await fetch(`/api/data/${this.selectedSymbol}`);
                    if (response.ok) {
                        const data = await response.json();
                        console.log('Refreshed data:', data);
                    }
                } catch (error) {
                    console.error('Error refreshing data:', error);
                }
            }
        }

        const analyzer = new OrderBookAnalyzer();
    </script>
</body>
</html>
'''

class OrderBookAnalyzer:
    def __init__(self):
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSD']
        self.order_books = {}
        self.predictions = {}
        self.feature_history = defaultdict(lambda: deque(maxlen=100))
        self.models = {}
        self.scalers = {}
        self.running = False
        
        # Analysis parameters
        self.depth_levels = 5
        self.imbalance_threshold = 0.6
        self.fee_rate = 0.001

    def fetch_symbols(self) -> List[str]:
        """Fetch symbols from Binance.US (simplified for Render)"""
        try:
            response = requests.get('https://api.binance.us/api/v3/exchangeInfo', timeout=10)
            data = response.json()
            
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] in ['USDT', 'USD'] and
                    any(base in symbol_info['symbol'] for base in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'])):
                    symbols.append(symbol_info['symbol'])
            
            return symbols[:10]  # Limit for performance
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return self.symbols  # Fallback

    def fetch_orderbook_snapshot(self, symbol: str) -> Optional[Dict]:
        """Fetch order book snapshot from Binance.US"""
        try:
            url = f'https://api.binance.us/api/v3/depth?symbol={symbol}&limit=10'
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                    'asks': [[float(price), float(qty)] for price, qty in data['asks']],
                    'timestamp': time.time()
                }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
        return None

    def calculate_order_book_features(self, orderbook: Dict) -> Dict:
        """Calculate order book features"""
        try:
            bids = np.array(orderbook['bids'][:self.depth_levels])
            asks = np.array(orderbook['asks'][:self.depth_levels])
            
            if len(bids) == 0 or len(asks) == 0:
                return {}
            
            # Basic calculations
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Mid price and spread
            mid_price = (bids[0, 0] + asks[0, 0]) / 2
            spread = asks[0, 0] - bids[0, 0]
            spread_pct = spread / mid_price * 100
            
            # Weighted imbalance
            bid_weights = 1 / (1 + np.abs(bids[:, 0] - mid_price) / mid_price)
            ask_weights = 1 / (1 + np.abs(asks[:, 0] - mid_price) / mid_price)
            
            weighted_bid_volume = np.sum(bids[:, 1] * bid_weights)
            weighted_ask_volume = np.sum(asks[:, 1] * ask_weights)
            weighted_total = weighted_bid_volume + weighted_ask_volume
            weighted_imbalance = ((weighted_bid_volume - weighted_ask_volume) / 
                                weighted_total if weighted_total > 0 else 0)
            
            return {
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'imbalance': imbalance,
                'weighted_imbalance': weighted_imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'timestamp': orderbook['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return {}

    def predict_price_direction(self, symbol: str, features: Dict) -> Dict:
        """Simple prediction based on imbalance"""
        try:
            imbalance = features.get('weighted_imbalance', 0)
            
            # Simple threshold-based prediction
            if abs(imbalance) > self.imbalance_threshold:
                prediction = 1 if imbalance > 0 else 0
                probability = min(0.8, 0.5 + abs(imbalance) * 0.5)
            else:
                prediction = 0
                probability = 0.5
            
            confidence = 'high' if abs(imbalance) > 0.7 else 'medium' if abs(imbalance) > 0.4 else 'low'
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': confidence,
                'signal_strength': abs(imbalance)
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return {'prediction': 0, 'probability': 0.5, 'confidence': 'low'}

    def generate_trading_signal(self, symbol: str, features: Dict, prediction: Dict) -> Dict:
        """Generate trading signal"""
        try:
            imbalance = features.get('weighted_imbalance', 0)
            signal_strength = abs(imbalance)
            prediction_prob = prediction.get('probability', 0.5)
            
            strong_signal = (signal_strength > self.imbalance_threshold and prediction_prob > 0.65)
            
            if strong_signal:
                direction = 'BUY' if imbalance > 0 else 'SELL'
            else:
                direction = 'HOLD'
            
            return {
                'signal': direction,
                'strength': signal_strength,
                'strong_signal': strong_signal,
                'expected_return': signal_strength * 50 - self.fee_rate * 100
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'strong_signal': False}

    def process_orderbook(self, symbol: str, orderbook: Dict):
        """Process order book and emit updates"""
        try:
            features = self.calculate_order_book_features(orderbook)
            if not features:
                return
                
            self.feature_history[symbol].append(features)
            
            prediction = self.predict_price_direction(symbol, features)
            signal = self.generate_trading_signal(symbol, features, prediction)
            
            self.predictions[symbol] = {
                **features,
                **prediction,
                **signal,
                'timestamp': time.time()
            }
            
            # Emit to frontend
            socketio.emit('orderbook_update', {
                'symbol': symbol,
                'data': self.predictions[symbol],
                'orderbook': {
                    'bids': orderbook['bids'][:5],
                    'asks': orderbook['asks'][:5]
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing orderbook for {symbol}: {e}")

    def start_data_collection(self):
        """Start data collection loop"""
        self.running = True
        self.symbols = self.fetch_symbols()
        logger.info(f"Starting data collection for {len(self.symbols)} symbols")
        
        def collection_loop():
            while self.running:
                try:
                    for symbol in self.symbols:
                        if not self.running:
                            break
                            
                        orderbook = self.fetch_orderbook_snapshot(symbol)
                        if orderbook:
                            self.order_books[symbol] = orderbook
                            self.process_orderbook(symbol, orderbook)
                        
                        time.sleep(0.5)  # Small delay between symbols
                    
                    time.sleep(2)  # Main loop delay
                    
                except Exception as e:
                    logger.error(f"Error in collection loop: {e}")
                    time.sleep(5)
        
        threading.Thread(target=collection_loop, daemon=True).start()

    def stop(self):
        """Stop data collection"""
        self.running = False

# Global analyzer instance
analyzer = OrderBookAnalyzer()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/symbols')
def get_symbols():
    """Get all tracked symbols"""
    return jsonify({
        'symbols': analyzer.symbols,
        'count': len(analyzer.symbols)
    })

@app.route('/api/data/<symbol>')
def get_symbol_data(symbol):
    """Get latest data for a symbol"""
    if symbol in analyzer.predictions:
        return jsonify(analyzer.predictions[symbol])
    return jsonify({'error': 'Symbol not found'}), 404

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    return jsonify({
        'total_symbols': len(analyzer.symbols),
        'active_predictions': len(analyzer.predictions)
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to Order Book analyzer'})
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

# Start the analyzer when the module is imported
analyzer.start_data_collection()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
