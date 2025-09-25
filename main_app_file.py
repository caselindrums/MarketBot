#!/usr/bin/env python3
"""
Advanced Order Book Analysis Bot for Binance.US
Python 3.13 compatible version with threading async mode
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
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

# Flask app setup - Using threading for Python 3.13 compatibility
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize SocketIO with threading mode (compatible with Python 3.13)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

# HTML template embedded for Render deployment
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Order Book Analysis | Binance.US</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
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
            padding: 12px 18px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .status {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
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
            height: 350px;
            margin: 20px 0;
            position: relative;
        }
        .symbol-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .symbol-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
            border-left: 4px solid #a0aec0;
        }
        .symbol-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
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
            font-size: 1.3em;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 12px;
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
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
            font-size: 0.95em;
        }
        .signal {
            text-align: center;
            margin-top: 15px;
            padding: 10px;
            border-radius: 8px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
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
        select {
            padding: 10px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            background: white;
        }
        .performance-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: 700;
            color: #2d3748;
        }
        .stat-label {
            font-size: 0.8em;
            color: #718096;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Advanced Order Book Analysis</h1>
            <p>Real-time market microstructure analysis with ML-based price prediction</p>
            
            <div class="controls">
                <select id="symbolSelect">
                    <option value="">Loading symbols...</option>
                </select>
                <button class="btn" id="refreshBtn">🔄 Refresh Data</button>
                <button class="btn" id="clearBtn">🗑️ Clear Charts</button>
                <div class="status" id="connectionStatus">
                    <span>🔴</span> Disconnected
                </div>
            </div>
        </div>

        <div class="dashboard">
            <div class="card">
                <h3>📊 Order Book Depth</h3>
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
            <h3>🎯 Performance Dashboard</h3>
            <div class="performance-stats">
                <div class="stat-card">
                    <div class="stat-value" id="totalSymbols">0</div>
                    <div class="stat-label">Active Symbols</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalSignals">0</div>
                    <div class="stat-label">Total Signals</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="strongSignals">0</div>
                    <div class="stat-label">Strong Signals</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avgAccuracy">0%</div>
                    <div class="stat-label">Avg Confidence</div>
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
                this.socket = io({
                    transports: ['websocket', 'polling'],
                    timeout: 20000,
                    forceNew: true
                });
                this.selectedSymbol = 'BTCUSDT';
                this.symbolData = new Map();
                this.charts = {};
                this.isConnected = false;
                
                this.initializeSocketHandlers();
                this.initializeUI();
                this.initializeCharts();
                this.loadSymbols();
            }

            initializeSocketHandlers() {
                this.socket.on('connect', () => {
                    console.log('Connected to server');
                    this.isConnected = true;
                    this.updateConnectionStatus(true);
                });

                this.socket.on('disconnect', () => {
                    console.log('Disconnected from server');
                    this.isConnected = false;
                    this.updateConnectionStatus(false);
                });

                this.socket.on('orderbook_update', (data) => {
                    this.handleOrderBookUpdate(data);
                });

                this.socket.on('connect_error', (error) => {
                    console.error('Connection error:', error);
                    this.updateConnectionStatus(false);
                });
            }

            initializeUI() {
                document.getElementById('symbolSelect').addEventListener('change', (e) => {
                    if (e.target.value) {
                        this.selectedSymbol = e.target.value;
                        this.updateSelectedSymbol();
                    }
                });

                document.getElementById('refreshBtn').addEventListener('click', () => {
                    this.refreshData();
                });

                document.getElementById('clearBtn').addEventListener('click', () => {
                    this.clearCharts();
                });
            }

            initializeCharts() {
                // Order book chart
                const orderbookCtx = document.getElementById('orderbookChart').getContext('2d');
                this.charts.orderbook = new Chart(orderbookCtx, {
                    type: 'bar',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Bids',
                            data: [],
                            backgroundColor: 'rgba(72, 187, 120, 0.8)',
                            borderColor: 'rgba(72, 187, 120, 1)',
                            borderWidth: 1
                        }, {
                            label: 'Asks',
                            data: [],
                            backgroundColor: 'rgba(245, 101, 101, 0.8)',
                            borderColor: 'rgba(245, 101, 101, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Price Level'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Volume'
                                }
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Order Book Depth'
                            }
                        }
                    }
                });

                // Trends chart
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
                            yAxisID: 'y',
                            tension: 0.4,
                            fill: true
                        }, {
                            label: 'Imbalance',
                            data: [],
                            borderColor: 'rgba(237, 137, 54, 1)',
                            backgroundColor: 'rgba(237, 137, 54, 0.1)',
                            yAxisID: 'y1',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            mode: 'index',
                            intersect: false,
                        },
                        scales: {
                            x: {
                                display: true,
                                title: {
                                    display: true,
                                    text: 'Time'
                                }
                            },
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: {
                                    display: true,
                                    text: 'Price ($)'
                                }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                title: {
                                    display: true,
                                    text: 'Imbalance'
                                },
                                min: -1,
                                max: 1,
                                grid: {
                                    drawOnChartArea: false,
                                }
                            }
                        },
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
                    select.innerHTML = '<option value="">Select Symbol</option>';
                    
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

                this.updatePerformanceStats();
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
                card.querySelector('.spread-value').textContent = `${data.spread_pct?.toFixed(3) || 'N/A'}%`;
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
                            <div class="metric-label">Spread</div>
                            <div class="metric-value spread-value">0%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Prediction</div>
                            <div class="metric-value prediction-value">50%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value confidence-value">Low</div>
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

                // Update order book chart
                if (orderbook && orderbook.bids && orderbook.asks) {
                    this.updateOrderBookChart(orderbook);
                }
            }

            updateOrderBookChart(orderbook) {
                const orderbookChart = this.charts.orderbook;
                
                const bids = orderbook.bids.slice(0, 5);
                const asks = orderbook.asks.slice(0, 5);
                
                const labels = [...bids.map(b => b[0].toFixed(4)), ...asks.map(a => a[0].toFixed(4))];
                const bidVolumes = [...bids.map(b => b[1]), ...new Array(asks.length).fill(0)];
                const askVolumes = [...new Array(bids.length).fill(0), ...asks.map(a => a[1])];
                
                orderbookChart.data.labels = labels;
                orderbookChart.data.datasets[0].data = bidVolumes;
                orderbookChart.data.datasets[1].data = askVolumes;
                
                orderbookChart.update('none');
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

            updatePerformanceStats() {
                let totalSymbols = this.symbolData.size;
                let totalSignals = 0;
                let strongSignals = 0;
                let totalConfidence = 0;
                let validPredictions = 0;

                for (const [symbol, data] of this.symbolData) {
                    if (data.latest) {
                        totalSignals++;
                        if (data.latest.strong_signal) {
                            strongSignals++;
                        }
                        if (data.latest.probability && data.latest.probability > 0.5) {
                            totalConfidence += data.latest.probability;
                            validPredictions++;
                        }
                    }
                }

                const avgConfidence = validPredictions > 0 ? (totalConfidence / validPredictions * 100) : 0;

                document.getElementById('totalSymbols').textContent = totalSymbols;
                document.getElementById('totalSignals').textContent = totalSignals;
                document.getElementById('strongSignals').textContent = strongSignals;
                document.getElementById('avgAccuracy').textContent = `${avgConfidence.toFixed(1)}%`;
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

            clearCharts() {
                Object.values(this.charts).forEach(chart => {
                    chart.data.labels = [];
                    chart.data.datasets.forEach(dataset => {
                        dataset.data = [];
                    });
                    chart.update();
                });
            }

            async refreshData() {
                if (!this.selectedSymbol) return;
                
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
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSD',
            'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'UNIUSDT', 'LTCUSDT'
        ]
        self.order_books = {}
        self.predictions = {}
        self.feature_history = defaultdict(lambda: deque(maxlen=100))
        self.running = False
        self.lock = threading.Lock()
        
        # Analysis parameters
        self.depth_levels = 5
        self.imbalance_threshold = 0.6
        self.fee_rate = 0.001
        
        # Request session with timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'OrderBookAnalyzer/1.0'})

    def fetch_symbols(self) -> List[str]:
        """Fetch available symbols from Binance.US"""
        try:
            response = self.session.get(
                'https://api.binance.us/api/v3/exchangeInfo', 
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['quoteAsset'] in ['USDT', 'USD']):
                    symbols.append(symbol_info['symbol'])
            
            # Filter for major cryptocurrencies and limit for performance
            major_symbols = [s for s in symbols if any(
                base in s for base in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'LTC']
            )]
            
            return major_symbols[:15] if major_symbols else self.symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return self.symbols

    def fetch_orderbook_snapshot(self, symbol: str) -> Optional[Dict]:
        """Fetch order book snapshot"""
        try:
            url = f'https://api.binance.us/api/v3/depth?symbol={symbol}&limit=20'
            response = self.session.get(url, timeout=8)
            response.raise_for_status()
            
            data = response.json()
            return {
                'symbol': symbol,
                'bids': [[float(price), float(qty)] for price, qty in data['bids'][:10]],
                'asks': [[float(price), float(qty)] for price, qty in data['asks'][:10]],
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
        return None

    def calculate_order_book_features(self, orderbook: Dict) -> Dict:
        """Calculate comprehensive order book features"""
        try:
            bids = np.array(orderbook['bids'][:self.depth_levels])
            asks = np.array(orderbook['asks'][:self.depth_levels])
            
            if len(bids) == 0 or len(asks) == 0:
                return {}
            
            # Basic volume calculations
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            total_volume = bid_volume + ask_volume
            
            # Price calculations
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = spread / mid_price * 100
            
            # Order imbalance
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Weighted imbalance (distance-weighted)
            bid_weights = 1 / (1 + np.abs(bids[:, 0] - mid_price) / mid_price)
            ask_weights = 1 / (1 + np.abs(asks[:, 0] - mid_price) / mid_price)
            
            weighted_bid_volume = np.sum(bids[:, 1] * bid_weights)
            weighted_ask_volume = np.sum(asks[:, 1] * ask_weights)
            weighted_total = weighted_bid_volume + weighted_ask_volume
            weighted_imbalance = ((weighted_bid_volume - weighted_ask_volume) / 
                                weighted_total if weighted_total > 0 else 0)
            
            # Volume-weighted average prices
            bid_vwap = np.sum(bids[:, 0] * bids[:, 1]) / bid_volume if bid_volume > 0 else best_bid
            ask_vwap = np.sum(asks[:, 0] * asks[:, 1]) / ask_volume if ask_volume > 0 else best_ask
            
            return {
                'mid_price': float(mid_price),
                'spread': float(spread),
                'spread_pct': float(spread_pct),
                'imbalance': float(imbalance),
                'weighted_imbalance': float(weighted_imbalance),
                'bid_volume': float(bid_volume),
                'ask_volume': float(ask_volume),
                'bid_vwap': float(bid_vwap),
                'ask_vwap': float(ask_vwap),
                'timestamp': orderbook['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return {}

    def predict_price_direction(self, symbol: str, features: Dict) -> Dict:
        """Enhanced prediction based on multiple factors"""
        try:
            weighted_imbalance = features.get('weighted_imbalance', 0)
            spread_pct = features.get('spread_pct', 0)
            
            # Multi-factor prediction
            imbalance_signal = abs(weighted_imbalance)
            spread_signal = min(1.0, 1 / (1 + spread_pct)) if spread_pct > 0 else 0
            
            # Combined signal strength
            signal_strength = (imbalance_signal * 0.7) + (spread_signal * 0.3)
            
            # Prediction logic
            if signal_strength > self.imbalance_threshold:
                prediction = 1 if weighted_imbalance > 0 else 0
                probability = min(0.85, 0.5 + signal_strength * 0.4)
            else:
                prediction = 0
                probability = 0.5
            
            # Confidence levels
            if signal_strength > 0.75:
                confidence = 'high'
            elif signal_strength > 0.45:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence': confidence,
                'signal_strength': float(signal_strength)
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return {'prediction': 0, 'probability': 0.5, 'confidence': 'low', 'signal_strength': 0.0}

    def generate_trading_signal(self, symbol: str, features: Dict, prediction: Dict) -> Dict:
        """Generate comprehensive trading signal"""
        try:
            weighted_imbalance = features.get('weighted_imbalance', 0)
            signal_strength = prediction.get('signal_strength', 0)
            prediction_prob = prediction.get('probability', 0.5)
            spread_pct = features.get('spread_pct', 0)
            
            # Signal generation logic
            strong_signal = (signal_strength > self.imbalance_threshold and 
                           prediction_prob > 0.65 and 
                           spread_pct < 0.5)  # Low spread indicates good liquidity
            
            if strong_signal:
                direction = 'BUY' if weighted_imbalance > 0 else 'SELL'
            else:
                direction = 'HOLD'
            
            # Expected return calculation (simplified)
            expected_return = signal_strength * 75 - self.fee_rate * 100 - spread_pct * 10
            
            return {
                'signal': direction,
                'strength': float(signal_strength),
                'strong_signal': bool(strong_signal),
                'expected_return': float(expected_return)
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'strong_signal': False, 'expected_return': 0.0}

    def process_orderbook(self, symbol: str, orderbook: Dict):
        """Process order book and emit updates"""
        try:
            features = self.calculate_order_book_features(orderbook)
            if not features:
                return
                
            # Store feature history
            self.feature_history[symbol].append(features)
            
            # Generate prediction and signal
            prediction = self.predict_price_direction(symbol, features)
            signal = self.generate_trading_signal(symbol, features, prediction)
            
            # Combine all data
            combined_data = {
                **features,
                **prediction,
                **signal,
                'timestamp': time.time()
            }
            
            with self.lock:
                self.predictions[symbol] = combined_data
            
            # Emit to frontend with error handling
            try:
                socketio.emit('orderbook_update', {
                    'symbol': symbol,
                    'data': combined_data,
                    'orderbook': {
                        'bids': orderbook['bids'][:5],
                        'asks': orderbook['asks'][:5]
                    }
                })
            except Exception as emit_error:
                logger.error(f"Error emitting data for {symbol}: {emit_error}")
                
        except Exception as e:
            logger.error(f"Error processing orderbook for {symbol}: {e}")

    def start_data_collection(self):
        """Start the main data collection loop"""
        if self.running:
            return
            
        self.running = True
        
        # Fetch symbols on startup
        try:
            self.symbols = self.fetch_symbols()
            logger.info(f"Starting data collection for {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"Error fetching symbols, using defaults: {e}")
        
        def collection_loop():
            consecutive_errors = 0
            max_consecutive_errors = 10
            
            while self.running:
                try:
                    for symbol in self.symbols:
                        if not self.running:
                            break
                            
                        try:
                            orderbook = self.fetch_orderbook_snapshot(symbol)
                            if orderbook:
                                with self.lock:
                                    self.order_books[symbol] = orderbook
                                self.process_orderbook(symbol, orderbook)
                                consecutive_errors = 0  # Reset on success
                            else:
                                logger.warning(f"No orderbook data for {symbol}")
                                
                        except Exception as symbol_error:
                            logger.error(f"Error processing {symbol}: {symbol_error}")
                            consecutive_errors += 1
                            
                        # Small delay between symbols to avoid rate limiting
                        if self.running:
                            time.sleep(0.3)
                    
                    # Main loop delay
                    if self.running:
                        time.sleep(2.5)
                        
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error in collection loop (attempt {consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical("Too many consecutive errors, stopping collection")
                        self.running = False
                        break
                        
                    # Exponential backoff on errors
                    time.sleep(min(30, 2 ** consecutive_errors))
        
        # Start collection in daemon thread
        collection_thread = threading.Thread(target=collection_loop, daemon=True)
        collection_thread.start()
        logger.info("Data collection thread started")

    def stop(self):
        """Stop data collection gracefully"""
        logger.info("Stopping data collection...")
        self.running = False
        
        # Close session
        try:
            self.session.close()
        except:
            pass

    def get_status(self) -> Dict:
        """Get analyzer status"""
        with self.lock:
            return {
                'running': self.running,
                'total_symbols': len(self.symbols),
                'active_predictions': len(self.predictions),
                'symbols': self.symbols
            }

# Global analyzer instance
analyzer = OrderBookAnalyzer()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/symbols')
def get_symbols():
    """Get all tracked symbols"""
    status = analyzer.get_status()
    return jsonify({
        'symbols': status['symbols'],
        'count': status['total_symbols']
    })

@app.route('/api/data/<symbol>')
def get_symbol_data(symbol):
    """Get latest data for a symbol"""
    with analyzer.lock:
        if symbol in analyzer.predictions:
            return jsonify(analyzer.predictions[symbol])
    return jsonify({'error': 'Symbol not found'}), 404

@app.route('/api/performance')
def get_performance():
    """Get performance metrics"""
    status = analyzer.get_status()
    return jsonify(status)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'analyzer_running': analyzer.running,
        'timestamp': time.time()
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

@socketio.on_error_default
def default_error_handler(e):
    """Handle SocketIO errors"""
    logger.error(f"SocketIO error: {e}")

# Initialize analyzer on startup
def initialize_app():
    """Initialize the application"""
    try:
        analyzer.start_data_collection()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {e}")

# Register shutdown handler
import atexit
atexit.register(analyzer.stop)

# Start analyzer
initialize_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask-SocketIO server on port {port}")
    
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=port, 
        debug=debug,
        allow_unsafe_werkzeug=True
    )
