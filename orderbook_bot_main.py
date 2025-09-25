#!/usr/bin/env python3
"""
Advanced Order Book Analysis Bot for Binance.US
Real-time order book analysis with ML-based price prediction
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
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import threading
import websockets
from collections import deque, defaultdict
import traceback

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

class OrderBookAnalyzer:
    def __init__(self):
        self.symbols = []
        self.order_books = {}
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.imbalance_history = defaultdict(lambda: deque(maxlen=1000))
        self.feature_history = defaultdict(lambda: deque(maxlen=500))
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.last_update = {}
        self.websocket_tasks = {}
        self.running = False
        
        # Analysis parameters
        self.depth_levels = 10
        self.imbalance_threshold = 0.6
        self.prediction_window = 60  # seconds
        self.fee_rate = 0.001  # 0.1% round-trip
        
        # Performance tracking
        self.trade_signals = defaultdict(list)
        self.accuracy_scores = defaultdict(float)

    async def fetch_symbols(self) -> List[str]:
        """Fetch all active USDT pairs from Binance.US"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.us/api/v3/exchangeInfo') as response:
                    data = await response.json()
                    
                    symbols = []
                    for symbol_info in data['symbols']:
                        if (symbol_info['status'] == 'TRADING' and 
                            symbol_info['quoteAsset'] == 'USDT' and
                            symbol_info['symbol'].endswith('USDT')):
                            symbols.append(symbol_info['symbol'])
                    
                    # Focus on most liquid pairs for better analysis
                    priority_symbols = [s for s in symbols if any(base in s for base in 
                        ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'UNI', 'AVAX', 'MATIC'])]
                    
                    return priority_symbols[:20]  # Limit to top 20 for performance
                    
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # Fallback

    async def fetch_orderbook_snapshot(self, symbol: str) -> Optional[Dict]:
        """Fetch initial order book snapshot"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f'https://api.binance.us/api/v3/depth?symbol={symbol}&limit=20'
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
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
        """Calculate advanced order book features"""
        try:
            bids = np.array(orderbook['bids'][:self.depth_levels])
            asks = np.array(orderbook['asks'][:self.depth_levels])
            
            if len(bids) == 0 or len(asks) == 0:
                return {}
            
            # Basic imbalance
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Weighted imbalance (closer to mid price has more weight)
            mid_price = (bids[0, 0] + asks[0, 0]) / 2
            bid_weights = 1 / (1 + np.abs(bids[:, 0] - mid_price) / mid_price)
            ask_weights = 1 / (1 + np.abs(asks[:, 0] - mid_price) / mid_price)
            
            weighted_bid_volume = np.sum(bids[:, 1] * bid_weights)
            weighted_ask_volume = np.sum(asks[:, 1] * ask_weights)
            weighted_total = weighted_bid_volume + weighted_ask_volume
            weighted_imbalance = ((weighted_bid_volume - weighted_ask_volume) / 
                                weighted_total if weighted_total > 0 else 0)
            
            # Spread analysis
            spread = asks[0, 0] - bids[0, 0]
            spread_pct = spread / mid_price * 100
            
            # Depth analysis
            bid_depth_5 = np.sum(bids[:5, 1])
            ask_depth_5 = np.sum(asks[:5, 1])
            depth_imbalance = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5)
            
            # Price level concentration
            total_bid_volume = np.sum(bids[:, 1])
            total_ask_volume = np.sum(asks[:, 1])
            top_bid_concentration = bids[0, 1] / total_bid_volume if total_bid_volume > 0 else 0
            top_ask_concentration = asks[0, 1] / total_ask_volume if total_ask_volume > 0 else 0
            
            return {
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'imbalance': imbalance,
                'weighted_imbalance': weighted_imbalance,
                'depth_imbalance': depth_imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'top_bid_concentration': top_bid_concentration,
                'top_ask_concentration': top_ask_concentration,
                'timestamp': orderbook['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return {}

    def prepare_ml_features(self, symbol: str, current_features: Dict) -> Optional[np.ndarray]:
        """Prepare feature vector for ML prediction"""
        try:
            if len(self.feature_history[symbol]) < 10:
                return None
                
            # Current features
            features = [
                current_features.get('imbalance', 0),
                current_features.get('weighted_imbalance', 0),
                current_features.get('depth_imbalance', 0),
                current_features.get('spread_pct', 0),
                current_features.get('top_bid_concentration', 0),
                current_features.get('top_ask_concentration', 0)
            ]
            
            # Historical features (moving averages)
            recent_features = list(self.feature_history[symbol])[-10:]
            for feature_key in ['imbalance', 'weighted_imbalance', 'depth_imbalance']:
                values = [f.get(feature_key, 0) for f in recent_features if feature_key in f]
                if values:
                    features.extend([
                        np.mean(values),
                        np.std(values),
                        values[-1] - values[0] if len(values) > 1 else 0  # trend
                    ])
                else:
                    features.extend([0, 0, 0])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ML features for {symbol}: {e}")
            return None

    def train_model(self, symbol: str):
        """Train prediction model for a symbol"""
        try:
            if len(self.feature_history[symbol]) < 100 or len(self.price_history[symbol]) < 100:
                return
                
            # Prepare training data
            features_list = []
            labels = []
            
            feature_data = list(self.feature_history[symbol])
            price_data = list(self.price_history[symbol])
            
            for i in range(10, len(feature_data) - 5):  # Need history and future data
                # Features from current state
                current_features = self.prepare_ml_features(symbol, feature_data[i])
                if current_features is None:
                    continue
                    
                # Label: price direction in next 5 periods (simplified to binary)
                current_price = feature_data[i].get('mid_price', 0)
                future_price = feature_data[min(i + 5, len(feature_data) - 1)].get('mid_price', current_price)
                
                price_change_pct = (future_price - current_price) / current_price * 100
                label = 1 if price_change_pct > 0.02 else 0  # 2 bps threshold
                
                features_list.append(current_features[0])
                labels.append(label)
            
            if len(features_list) < 50:
                return
                
            X = np.array(features_list)
            y = np.array(labels)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(random_state=42, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            accuracy = model.score(X_test_scaled, y_test)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.accuracy_scores[symbol] = accuracy
            
            logger.info(f"Trained model for {symbol} with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")

    def predict_price_direction(self, symbol: str, features: Dict) -> Dict:
        """Predict price direction using trained model"""
        try:
            if symbol not in self.models or symbol not in self.scalers:
                return {'prediction': 0, 'probability': 0.5, 'confidence': 'low'}
                
            X = self.prepare_ml_features(symbol, features)
            if X is None:
                return {'prediction': 0, 'probability': 0.5, 'confidence': 'low'}
                
            X_scaled = self.scalers[symbol].transform(X)
            prediction = self.models[symbol].predict(X_scaled)[0]
            probabilities = self.models[symbol].predict_proba(X_scaled)[0]
            
            confidence = 'high' if max(probabilities) > 0.7 else 'medium' if max(probabilities) > 0.6 else 'low'
            
            return {
                'prediction': int(prediction),
                'probability': float(max(probabilities)),
                'confidence': confidence,
                'signal_strength': abs(features.get('weighted_imbalance', 0))
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return {'prediction': 0, 'probability': 0.5, 'confidence': 'low'}

    async def process_orderbook(self, symbol: str, orderbook: Dict):
        """Process order book update and generate signals"""
        try:
            # Calculate features
            features = self.calculate_order_book_features(orderbook)
            if not features:
                return
                
            # Store historical data
            self.feature_history[symbol].append(features)
            self.price_history[symbol].append(features['mid_price'])
            self.last_update[symbol] = time.time()
            
            # Train model periodically
            if len(self.feature_history[symbol]) % 50 == 0:
                threading.Thread(target=self.train_model, args=(symbol,)).start()
            
            # Generate prediction
            prediction = self.predict_price_direction(symbol, features)
            
            # Generate trading signal
            signal = self.generate_trading_signal(symbol, features, prediction)
            
            # Store results
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

    def generate_trading_signal(self, symbol: str, features: Dict, prediction: Dict) -> Dict:
        """Generate trading signal based on analysis"""
        try:
            imbalance = features.get('weighted_imbalance', 0)
            signal_strength = abs(imbalance)
            prediction_prob = prediction.get('probability', 0.5)
            
            # Signal conditions
            strong_signal = (signal_strength > self.imbalance_threshold and 
                           prediction_prob > 0.65)
            
            direction = 'BUY' if imbalance > 0 and prediction['prediction'] == 1 else \
                       'SELL' if imbalance < 0 and prediction['prediction'] == 0 else 'HOLD'
            
            # Calculate potential profit (simplified)
            spread_pct = features.get('spread_pct', 0)
            expected_return = signal_strength * 100 - self.fee_rate * 100 - spread_pct
            
            return {
                'signal': direction,
                'strength': signal_strength,
                'strong_signal': strong_signal,
                'expected_return': expected_return,
                'risk_score': spread_pct + (1 - prediction_prob) * 50
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'strong_signal': False}

    async def websocket_handler(self, symbol: str):
        """Handle WebSocket connection for a symbol"""
        ws_url = f"wss://stream.binance.us:9443/ws/{symbol.lower()}@depth"
        
        while self.running:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"Connected to WebSocket for {symbol}")
                    
                    while self.running:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                            data = json.loads(message)
                            
                            # Update order book with incremental data
                            if symbol in self.order_books:
                                # For simplicity, we'll fetch fresh snapshots periodically
                                # In production, you'd want to apply incremental updates properly
                                continue
                                
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            await websocket.ping()
                            continue
                            
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting

    async def start_data_collection(self):
        """Start collecting data for all symbols"""
        self.running = True
        self.symbols = await self.fetch_symbols()
        logger.info(f"Starting data collection for {len(self.symbols)} symbols")
        
        # Fetch initial snapshots
        for symbol in self.symbols:
            orderbook = await self.fetch_orderbook_snapshot(symbol)
            if orderbook:
                self.order_books[symbol] = orderbook
                await self.process_orderbook(symbol, orderbook)
        
        # Start periodic snapshot updates (every 5 seconds)
        asyncio.create_task(self.periodic_updates())
        
        # Start WebSocket connections (for real-time updates in production)
        # for symbol in self.symbols:
        #     self.websocket_tasks[symbol] = asyncio.create_task(
        #         self.websocket_handler(symbol)
        #     )

    async def periodic_updates(self):
        """Periodically fetch fresh order book snapshots"""
        while self.running:
            try:
                tasks = []
                for symbol in self.symbols:
                    tasks.append(self.fetch_orderbook_snapshot(symbol))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, result in zip(self.symbols, results):
                    if isinstance(result, dict) and result:
                        self.order_books[symbol] = result
                        await self.process_orderbook(symbol, result)
                
                await asyncio.sleep(3)  # Update every 3 seconds
                
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """Stop data collection"""
        self.running = False
        for task in self.websocket_tasks.values():
            task.cancel()

# Global analyzer instance
analyzer = OrderBookAnalyzer()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

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
    """Get model performance metrics"""
    return jsonify({
        'accuracy_scores': dict(analyzer.accuracy_scores),
        'total_symbols': len(analyzer.symbols),
        'active_models': len(analyzer.models)
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

def run_async_loop():
    """Run the async event loop in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(analyzer.start_data_collection())

if __name__ == '__main__':
    # Start the async data collection in a separate thread
    threading.Thread(target=run_async_loop, daemon=True).start()
    
    # Start Flask-SocketIO server
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
