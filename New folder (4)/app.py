from flask import Flask, render_template, request, jsonify, session
from flask_pymongo import PyMongo
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import pytz
import json
from bson import json_util
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import random
from bson import ObjectId
import google.generativeai as genai
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'  # Add this for session management

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/energy_monitor"
mongo = PyMongo(app)

# Ensure required collections exist
try:
    db = mongo.db
    energy_data = db.energy_usage
    food_items = db.food_items
    deliveries = db.deliveries
    impact_stats = db.impact_stats
    readings = db.readings  # Add readings collection
    food_waste = db.food_waste  # Add food_waste collection
    waste_stats = db.waste_stats  # Add waste_stats collection
    carbon_activities = db.carbon_activities  # Add carbon_activities collection
except Exception as e:
    print(f"MongoDB connection error: {e}")

# Appliance power ranges in kWh per hour
APPLIANCE_POWER = {
    'AC': {'min': 1.0, 'max': 4.0, 'typical': 2.5},
    'Refrigerator': {'min': 0.1, 'max': 0.5, 'typical': 0.2},
    'Washing Machine': {'min': 0.4, 'max': 1.5, 'typical': 0.8},
    'TV': {'min': 0.1, 'max': 0.4, 'typical': 0.2},
    'Lights': {'min': 0.02, 'max': 0.2, 'typical': 0.06},
    'Computer': {'min': 0.1, 'max': 0.5, 'typical': 0.2},
    'Microwave': {'min': 0.6, 'max': 1.5, 'typical': 1.0},
    'Water Heater': {'min': 1.5, 'max': 4.0, 'typical': 2.5},
    'Fan': {'min': 0.05, 'max': 0.2, 'typical': 0.1},
    'Other': {'min': 0.05, 'max': 3.0, 'typical': 0.5}
}

# Set timezone to IST
IST = pytz.timezone('Asia/Kolkata')

def get_current_time():
    """Get current time in IST"""
    return datetime.now(IST)

def generate_sample_data():
    """Generate 24 hours of sample data if database is empty"""
    try:
        # Clear existing data for testing
        energy_data.delete_many({})
        
        current_time = get_current_time()
        sample_data = []
        
        # Generate data for the last 24 hours
        for hour in range(24):
            time_point = current_time - timedelta(hours=hour)
            
            # Generate readings for different appliances
            for appliance in APPLIANCE_POWER.keys():
                # More usage during peak hours (7-9 AM and 6-10 PM)
                is_peak = time_point.hour in [7, 8, 9, 18, 19, 20, 21, 22]
                multiplier = 1.5 if is_peak else 1.0
                
                power_range = APPLIANCE_POWER[appliance]
                base_usage = power_range['typical']
                variation = (power_range['max'] - power_range['min']) * 0.2
                usage = base_usage + random.uniform(-variation, variation)
                usage = usage * multiplier
                
                # Ensure usage stays within defined ranges
                usage = max(power_range['min'], min(power_range['max'], usage))
                
                reading = {
                    "timestamp": time_point,
                    "appliance": appliance,
                    "usage": round(usage, 3),
                    "cost": round(usage * 0.12, 2)  # $0.12 per kWh
                }
                sample_data.append(reading)
        
        if sample_data:
            energy_data.insert_many(sample_data)
            print(f"Generated {len(sample_data)} sample readings")
            
            # Verify data was inserted
            count = energy_data.count_documents({})
            print(f"Total documents in database: {count}")
            
            # Print a few sample readings
            print("\nSample readings:")
            for reading in energy_data.find().limit(3):
                print(f"Time: {reading['timestamp']}, Appliance: {reading['appliance']}, Usage: {reading['usage']} kWh/h")
        
    except Exception as e:
        print(f"Error generating sample data: {e}")

# Generate sample data on startup
generate_sample_data()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set!")
    GEMINI_API_KEY = 'AIzaSyCANIGZZ6jEIdpekln-TrA7BDbfoaPxPEg'  # Fallback for development

genai.configure(api_key=GEMINI_API_KEY)

# List and print available models
try:
    available_models = genai.list_models()
    print("\nAvailable Gemini models:")
    for model in available_models:
        print(f"- {model.name} ({model.supported_generation_methods})")
except Exception as e:
    print(f"Error listing Gemini models: {e}")

# Initialize Gemini model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Successfully initialized Gemini model")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")

# Main routes
@app.route('/')
def index():
    # Initialize user stats in session if not present
    if 'user_stats' not in session:
        try:
            # Try to get existing stats from database
            user_stats = mongo.db.waste_stats.find_one({'user_id': 'default_user'})
            if user_stats:
                session['user_stats'] = {
                    'items_recycled': user_stats.get('items_recycled', 0),
                    'eco_points': user_stats.get('eco_points', 0),
                    'streak': user_stats.get('streak', 0)
                }
            else:
                # Initialize new user stats
                session['user_stats'] = {
                    'items_recycled': 0,
                    'eco_points': 0,
                    'streak': 0
                }
        except Exception as e:
            print(f"Error initializing stats: {str(e)}")
            session['user_stats'] = {
                'items_recycled': 0,
                'eco_points': 0,
                'streak': 0
            }
    return render_template('index.html', user_stats=session.get('user_stats', {}))

@app.route('/food_waste')
def food_waste_page():
    return render_template('food_waste.html')

@app.route('/waste_segregation')
def waste_segregation():
    # Get current stats from session
    user_stats = session.get('user_stats', {
        'items_recycled': 0,
        'eco_points': 0,
        'streak': 0
    })
    return render_template('waste_segregation.html', user_stats=user_stats)

# Energy monitoring routes
@app.route('/get_usage_data')
def get_usage_data():
    try:
        # Get recent readings
        recent_readings = list(readings.find().sort('timestamp', -1).limit(10))
        
        # Convert ObjectId to string for JSON serialization
        for reading in recent_readings:
            reading['_id'] = str(reading['_id'])
            reading['timestamp'] = reading['timestamp'].strftime('%I:%M %p')
            reading['cost'] = float(reading['usage']) * 0.12  # Example rate of $0.12 per kWh
            reading['trend'] = 'up' if float(reading['usage']) > 1.0 else 'down'
        
        # Calculate metrics
        current_usage = float(recent_readings[0]['usage']) if recent_readings else 0
        total_cost = sum(float(r['usage']) * 0.12 for r in recent_readings)
        
        # Get hourly data for chart
        hourly_data = list(readings.find().sort('timestamp', -1).limit(24))
        
        # Process chart data
        chart_data = {
            'labels': [],
            'values': []
        }
        
        if hourly_data:
            chart_data['labels'] = [h['timestamp'].strftime('%I:%M %p') for h in hourly_data][::-1]
            chart_data['values'] = [float(h['usage']) for h in hourly_data][::-1]
        
        # Calculate peak hour
        if hourly_data:
            peak_reading = max(hourly_data, key=lambda x: float(x['usage']))
            peak_hour = peak_reading['timestamp'].strftime('%I:%M %p')
        else:
            peak_hour = '00:00'
        
        # Calculate predicted usage (average of recent readings)
        if recent_readings:
            predicted_usage = sum(float(r['usage']) for r in recent_readings) / len(recent_readings)
        else:
            predicted_usage = 0
        
        metrics = {
            'current_usage': current_usage,
            'total_cost': total_cost,
            'peak_hour': peak_hour,
            'predicted_usage': predicted_usage
        }
        
        return jsonify({
            'recent_readings': recent_readings,
            'chart_data': chart_data,
            'metrics': metrics
        })
    except Exception as e:
        print(f"Error in get_usage_data: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/add_usage', methods=['POST'])
def add_usage():
    try:
        data = request.json
        appliance = data.get('appliance')
        usage = float(data.get('usage'))
        
        # Validate the reading
        if not appliance or usage < 0:
            return jsonify({'success': False, 'error': 'Invalid input data'}), 400
        
        # Add the reading to database
        reading = {
            'appliance': appliance,
            'usage': usage,
            'timestamp': datetime.now(),
            'cost': usage * 0.12  # Example rate of $0.12 per kWh
        }
        
        readings.insert_one(reading)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error in add_usage: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_usage')
def get_usage():
    try:
        # Get last 24 hours of data
        start_time = get_current_time() - timedelta(hours=24)
        usage_data = list(energy_data.find(
            {"timestamp": {"$gte": start_time}},
            {"_id": 0, "timestamp": 1, "usage": 1, "cost": 1, "appliance": 1}
        ).sort("timestamp", 1))  # Sort by time ascending
        
        # Group data by hour for the chart
        hourly_data = {}
        for reading in usage_data:
            hour = reading['timestamp'].replace(minute=0, second=0, microsecond=0)
            if hour not in hourly_data:
                hourly_data[hour] = {
                    'timestamp': hour,
                    'usage': 0,
                    'cost': 0
                }
            hourly_data[hour]['usage'] += reading['usage']
            hourly_data[hour]['cost'] += reading['cost']
        
        # Convert to list and sort
        hourly_list = list(hourly_data.values())
        hourly_list.sort(key=lambda x: x['timestamp'])
        
        return json.dumps(hourly_list, default=json_util.default)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_summary')
def get_summary():
    try:
        # Get today's data (in IST)
        today_start = get_current_time().replace(hour=0, minute=0, second=0, microsecond=0)
        today_data = list(energy_data.find({"timestamp": {"$gte": today_start}}))
        
        # Get recent readings (last 10 readings)
        recent_readings = list(energy_data.find().sort("timestamp", -1).limit(10))
        
        if not today_data and not recent_readings:
            return jsonify({
                "total_usage": 0,
                "total_cost": 0,
                "peak_hour": None,
                "peak_hour_usage": 0,
                "avg_usage": 0,
                "appliance_breakdown": {},
                "recent_readings": []
            })
        
        # Calculate summary statistics
        total_usage = sum(d["usage"] for d in today_data)
        total_cost = sum(d["cost"] for d in today_data)
        avg_usage = total_usage / len(today_data) if today_data else 0
        
        # Calculate appliance breakdown
        appliance_usage = {}
        for d in today_data:
            app = d["appliance"]
            appliance_usage[app] = appliance_usage.get(app, 0) + d["usage"]
        
        # Find peak hour (group by hour and find max)
        peak_data = {"hour": None, "usage": 0}
        hour_usage = {}
        
        for reading in today_data:
            hour = reading["timestamp"].replace(minute=0, second=0, microsecond=0)
            if hour not in hour_usage:
                hour_usage[hour] = 0
            hour_usage[hour] += reading["usage"]
            
            if hour_usage[hour] > peak_data["usage"]:
                peak_data = {
                    "hour": hour,
                    "usage": hour_usage[hour]
                }
        
        # Calculate trend for each reading
        for i, reading in enumerate(recent_readings):
            if i < len(recent_readings) - 1:
                prev_usage = recent_readings[i + 1]["usage"]
                reading["trend"] = "up" if reading["usage"] > prev_usage else "down"
            else:
                reading["trend"] = "same"
        
        return jsonify({
            "total_usage": round(total_usage, 2),
            "total_cost": round(total_cost, 2),
            "peak_hour": peak_data["hour"],
            "peak_hour_usage": round(peak_data["usage"], 2),
            "avg_usage": round(avg_usage, 2),
            "appliance_breakdown": appliance_usage,
            "recent_readings": json.loads(json_util.dumps(recent_readings))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_appliance_ranges')
def get_appliance_ranges():
    """Return the typical ranges for all appliances"""
    appliances_with_icons = {
        'AC': {'icon': 'snowflake', 'ranges': APPLIANCE_POWER['AC']},
        'Refrigerator': {'icon': 'box', 'ranges': APPLIANCE_POWER['Refrigerator']},
        'Washing Machine': {'icon': 'washer', 'ranges': APPLIANCE_POWER['Washing Machine']},
        'TV': {'icon': 'tv', 'ranges': APPLIANCE_POWER['TV']},
        'Lights': {'icon': 'lightbulb', 'ranges': APPLIANCE_POWER['Lights']},
        'Computer': {'icon': 'laptop', 'ranges': APPLIANCE_POWER['Computer']},
        'Microwave': {'icon': 'microwave', 'ranges': APPLIANCE_POWER['Microwave']},
        'Water Heater': {'icon': 'hot-tub', 'ranges': APPLIANCE_POWER['Water Heater']},
        'Fan': {'icon': 'fan', 'ranges': APPLIANCE_POWER['Fan']},
        'Other': {'icon': 'plug', 'ranges': APPLIANCE_POWER['Other']}
    }
    return jsonify(appliances_with_icons)

def get_appliance_recommendations(usage_data):
    """Generate AI-driven recommendations based on usage patterns"""
    recommendations = []
    appliance_totals = {}
    
    # Calculate total usage per appliance
    for reading in usage_data:
        app = reading['appliance']
        if app not in appliance_totals:
            appliance_totals[app] = {
                'total': 0,
                'count': 0,
                'peak_usage': 0,
                'peak_time': None
            }
        
        appliance_totals[app]['total'] += reading['usage']
        appliance_totals[app]['count'] += 1
        
        if reading['usage'] > appliance_totals[app]['peak_usage']:
            appliance_totals[app]['peak_usage'] = reading['usage']
            appliance_totals[app]['peak_time'] = reading['timestamp']
    
    # Generate recommendations based on patterns
    for app, data in appliance_totals.items():
        avg_usage = data['total'] / data['count']
        typical = APPLIANCE_POWER[app]['typical']
        
        if avg_usage > typical * 1.2:  # Using 20% above typical as threshold
            if app == 'AC':
                recommendations.append({
                    'appliance': app,
                    'severity': 'high',
                    'tip': 'Consider setting AC temperature 1-2 degrees higher to save energy',
                    'saving_potential': f"{round((avg_usage - typical) * 0.12 * 24 * 30, 2)} per month"
                })
            elif app == 'Refrigerator':
                recommendations.append({
                    'appliance': app,
                    'severity': 'medium',
                    'tip': 'Check refrigerator door seal and avoid frequent door opening',
                    'saving_potential': f"{round((avg_usage - typical) * 0.12 * 24 * 30, 2)} per month"
                })
            elif app == 'Lights':
                recommendations.append({
                    'appliance': app,
                    'severity': 'low',
                    'tip': 'Consider switching to LED bulbs and utilizing natural light',
                    'saving_potential': f"{round((avg_usage - typical) * 0.12 * 24 * 30, 2)} per month"
                })
    
    return recommendations

def calculate_savings_potential(current_usage, recommendations):
    """Calculate potential savings from implementing recommendations"""
    monthly_savings = 0
    yearly_savings = 0
    
    for rec in recommendations:
        saving = float(rec['saving_potential'].split()[0])
        monthly_savings += saving
    
    yearly_savings = monthly_savings * 12
    
    return {
        'monthly': round(monthly_savings, 2),
        'yearly': round(yearly_savings, 2),
        'co2_reduction': round(yearly_savings * 0.85, 2)  # kg of CO2 per kWh saved
    }

@app.route('/get_insights')
def get_insights():
    try:
        # Get last 7 days of data for better pattern recognition
        start_time = get_current_time() - timedelta(days=7)
        usage_data = list(energy_data.find({"timestamp": {"$gte": start_time}}))
        
        if not usage_data:
            return jsonify({
                "recommendations": [],
                "savings_potential": {
                    "monthly": 0,
                    "yearly": 0,
                    "co2_reduction": 0
                },
                "predictions": {
                    "next_day": 0,
                    "next_week": 0,
                    "confidence": 0
                }
            })
        
        # Generate recommendations
        recommendations = get_appliance_recommendations(usage_data)
        
        # Calculate potential savings
        total_usage = sum(d["usage"] for d in usage_data)
        savings = calculate_savings_potential(total_usage, recommendations)
        
        # Prepare usage data for prediction
        df = pd.DataFrame(usage_data)
        df['hour'] = df['timestamp'].apply(lambda x: x.hour)
        df['day'] = df['timestamp'].apply(lambda x: x.weekday())
        
        # Train model on historical data
        X = df[['hour', 'day']]
        y = df['usage']
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next day and week
        next_24h = []
        current_hour = get_current_time().hour
        current_day = get_current_time().weekday()
        
        for h in range(24):
            next_hour = (current_hour + h) % 24
            next_day = (current_day + (current_hour + h) // 24) % 7
            prediction = model.predict([[next_hour, next_day]])[0]
            next_24h.append(max(0, prediction))
        
        confidence = max(0, min(100, model.score(X, y) * 100))
        
        return jsonify({
            "recommendations": recommendations,
            "savings_potential": savings,
            "predictions": {
                "next_day": round(sum(next_24h), 2),
                "next_week": round(sum(next_24h) * 7, 2),
                "confidence": round(confidence, 1),
                "hourly_forecast": [round(x, 2) for x in next_24h]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_usage')
def predict_usage():
    try:
        # Get historical data for prediction
        data = list(energy_data.find().sort("timestamp", -1).limit(24))
        
        if len(data) < 12:  # Need minimum data for prediction
            return jsonify({"error": "Insufficient data for prediction"}), 400
            
        df = pd.DataFrame(data)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Prepare data for model
        X = df[['hour']].values
        y = df['usage'].values
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next hour
        next_hour = (get_current_time().hour + 1) % 24
        prediction = model.predict([[next_hour]])[0]
        
        # Calculate confidence based on R² score
        confidence = model.score(X, y)
        
        return jsonify({
            "predicted_usage": round(prediction, 2),
            "confidence": round(confidence * 100, 1),
            "next_hour": next_hour
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/items', methods=['GET'])
def get_food_items():
    try:
        items = list(food_items.find({"expiry": {"$gt": get_current_time()}}))
        return jsonify(json.loads(json_util.dumps(items)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/items', methods=['POST'])
def add_food_item():
    try:
        data = request.json
        item = {
            "type": data['type'],
            "quantity": float(data['quantity']),
            "expiry": datetime.fromisoformat(data['expiry']),
            "storage": data['storage'],
            "added_at": get_current_time(),
            "status": "available"
        }
        result = food_items.insert_one(item)
        
        # Update impact stats
        impact_stats.update_one(
            {"_id": "global"},
            {
                "$inc": {
                    "total_food_saved": item['quantity'],
                    "meals_provided": int(item['quantity'] * 2),
                    "co2_saved": item['quantity'] * 2.5
                }
            },
            upsert=True
        )
        
        return jsonify({"success": True, "id": str(result.inserted_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/claim/<item_id>', methods=['POST'])
def claim_food_item():
    try:
        data = request.json
        item = food_items.find_one_and_update(
            {"_id": ObjectId(item_id), "status": "available"},
            {"$set": {"status": "claimed"}}
        )
        
        if not item:
            return jsonify({"error": "Item not found or already claimed"}), 404
        
        # Create delivery
        delivery = {
            "food_item_id": item_id,
            "from_location": data['from_location'],
            "to_location": data['to_location'],
            "status": "scheduled",
            "created_at": get_current_time(),
            "estimated_pickup": get_current_time() + timedelta(minutes=15),
            "estimated_delivery": get_current_time() + timedelta(minutes=45)
        }
        
        deliveries.insert_one(delivery)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/deliveries')
def get_deliveries():
    try:
        active_deliveries = list(deliveries.find({
            "status": {"$in": ["scheduled", "in_transit"]}
        }).sort("created_at", -1))
        return jsonify(json.loads(json_util.dumps(active_deliveries)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/impact')
def get_impact_stats():
    try:
        stats = impact_stats.find_one({"_id": "global"}) or {
            "total_food_saved": 0,
            "meals_provided": 0,
            "co2_saved": 0
        }
        return jsonify(json.loads(json_util.dumps(stats)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/food/update_delivery/<delivery_id>', methods=['POST'])
def update_delivery_status():
    try:
        data = request.json
        delivery = deliveries.find_one_and_update(
            {"_id": ObjectId(delivery_id)},
            {"$set": {
                "status": data['status'],
                "updated_at": get_current_time()
            }}
        )
        
        if not delivery:
            return jsonify({"error": "Delivery not found"}), 404
            
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_waste_data')
def get_waste_data():
    try:
        # Get recent records
        recent_records = list(food_waste.find().sort('timestamp', -1).limit(10))
        
        # Process records
        for record in recent_records:
            record['_id'] = str(record['_id'])
            record['date'] = record['timestamp'].strftime('%Y-%m-%d %I:%M %p')
            record['impact'] = record['quantity'] * 2.5  # 2.5 kg CO2 per kg of food waste
        
        # Get daily data for chart
        daily_data = list(food_waste.find({
            'timestamp': {
                '$gte': datetime.now() - timedelta(days=7)
            }
        }).sort('timestamp', 1))
        
        # Process chart data
        chart_data = {
            'labels': [],
            'values': []
        }
        
        if daily_data:
            # Group by date
            daily_totals = {}
            for record in daily_data:
                date = record['timestamp'].strftime('%Y-%m-%d')
                if date not in daily_totals:
                    daily_totals[date] = 0
                daily_totals[date] += record['quantity']
            
            # Sort by date
            sorted_dates = sorted(daily_totals.keys())
            chart_data['labels'] = [datetime.strptime(date, '%Y-%m-%d').strftime('%b %d') for date in sorted_dates]
            chart_data['values'] = [daily_totals[date] for date in sorted_dates]
        
        # Calculate metrics
        total_waste = sum(r['quantity'] for r in recent_records) if recent_records else 0
        cost_impact = total_waste * 5  # Example: $5 per kg of food waste
        environmental_impact = total_waste * 2.5  # 2.5 kg CO2 per kg of food waste
        
        # Calculate waste reduction (example: compare to previous period)
        previous_records = list(food_waste.find({
            'timestamp': {
                '$gte': datetime.now() - timedelta(days=14),
                '$lt': datetime.now() - timedelta(days=7)
            }
        }))
        
        previous_waste = sum(r['quantity'] for r in previous_records) if previous_records else 0
        current_waste = sum(r['quantity'] for r in daily_data) if daily_data else 0
        
        if previous_waste > 0:
            waste_reduction = ((previous_waste - current_waste) / previous_waste) * 100
        else:
            waste_reduction = 0
        
        metrics = {
            'total_waste': total_waste,
            'cost_impact': cost_impact,
            'environmental_impact': environmental_impact,
            'waste_reduction': waste_reduction
        }
        
        return jsonify({
            'recent_records': recent_records,
            'chart_data': chart_data,
            'metrics': metrics
        })
    except Exception as e:
        print(f"Error in get_waste_data: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/add_waste', methods=['POST'])
def add_waste():
    try:
        data = request.json
        food_type = data.get('food_type')
        quantity = float(data.get('quantity'))
        reason = data.get('reason')
        
        # Validate the data
        if not food_type or not reason or quantity <= 0:
            return jsonify({'success': False, 'error': 'Invalid input data'}), 400
        
        # Add the record to database
        record = {
            'food_type': food_type,
            'quantity': quantity,
            'reason': reason,
            'timestamp': datetime.now(),
            'impact': quantity * 2.5  # 2.5 kg CO2 per kg of food waste
        }
        
        food_waste.insert_one(record)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error in add_waste: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze_waste', methods=['POST'])
def analyze_waste():
    """Analyze waste image using Gemini API"""
    try:
        print("\n=== Starting waste analysis ===")
        # Get image from request
        if 'image' not in request.files:
            print("No image file in request")
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        if not image_file.filename:
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        # Print debug information
        print(f"Received file: {image_file.filename}")
        print(f"File content type: {image_file.content_type}")
        
        # Read and process image
        try:
            # Read image bytes first
            image_bytes = image_file.read()
            print(f"Read {len(image_bytes)} bytes from image file")
            
            # Create BytesIO object
            image_buffer = io.BytesIO(image_bytes)
            
            # Open image with PIL
            image = Image.open(image_buffer)
            print(f"Image opened successfully: format={image.format}, size={image.size}, mode={image.mode}")
            
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                print(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Ensure image is not too large (max 4MB after processing)
            max_size = (1024, 1024)  # Maximum dimensions
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                print(f"Resizing image from {image.size} to max dimensions {max_size}")
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to JPEG format in memory
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=85)
            output_buffer.seek(0)
            processed_image = output_buffer.getvalue()
            print(f"Processed image size: {len(processed_image)} bytes")
            
        except Exception as img_error:
            print(f"Error processing image: {str(img_error)}")
            return jsonify({'error': f'Invalid image format: {str(img_error)}'}), 400
        
        # Prepare prompt for Gemini
        prompt = """Analyze this image and classify the waste item. Respond ONLY with a valid JSON object in this exact format, nothing else:
{
    "item_type": "brief description of the item",
    "category": "one of: recyclable, compostable, or waste",
    "instructions": "brief disposal instructions",
    "impact": "brief environmental impact"
}"""
        
        try:
            # Generate response from Gemini
            print("\nSending request to Gemini API...")
            response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": processed_image}])
            
            # Get the response text
            response_text = response.text.strip()
            print(f"Received response from Gemini API: {response_text}")
            
            # Extract JSON from the response text
            try:
                # Find JSON object in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start == -1 or json_end == -1:
                    raise ValueError("No JSON object found in response")
                
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Validate and clean the result
                required_fields = ["item_type", "category", "instructions", "impact"]
                for field in required_fields:
                    if field not in result or not result[field]:
                        result[field] = "Not specified"
                
                # Normalize category
                result["category"] = result["category"].lower().strip()
                if result["category"] not in ["recyclable", "compostable", "waste"]:
                    result["category"] = "waste"
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing response: {str(e)}")
                # Fallback to default response
                result = {
                    "item_type": "Unknown item",
                    "category": "waste",
                    "instructions": "Please consult local waste management guidelines",
                    "impact": "Proper waste disposal helps protect our environment"
                }
            
            # Get current stats from session
            user_stats = session.get('user_stats', {
                'items_recycled': 0,
                'eco_points': 0,
                'streak': 0
            })
            
            # Calculate points
            points = {
                'recyclable': 15,
                'compostable': 12,
                'waste': 10
            }.get(result['category'], 10)
            
            # Update stats
            user_stats['items_recycled'] += 1
            user_stats['eco_points'] += points
            user_stats['streak'] += 1
            
            # Save to session
            session['user_stats'] = user_stats
            
            # Update database
            try:
                mongo.db.waste_stats.update_one(
                    {'user_id': 'default_user'},
                    {
                        '$set': {
                            'items_recycled': user_stats['items_recycled'],
                            'eco_points': user_stats['eco_points'],
                            'streak': user_stats['streak'],
                            'last_updated': datetime.now(IST)
                        }
                    },
                    upsert=True
                )
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")
            
            # Educational facts
            facts = [
                "Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
                "Glass bottles can be recycled endlessly without quality degradation.",
                "Plastic bags take 10-1000 years to decompose in landfills.",
                "Composting food waste reduces methane emissions from landfills.",
                "Recycling paper saves trees and reduces water pollution.",
                "E-waste recycling recovers valuable metals and prevents toxic pollution."
            ]
            
            # Prepare response
            response_data = {
                **result,
                'stats': user_stats,
                'points_earned': points,
                'educational_fact': random.choice(facts)
            }
            
            print(f"Sending successful response: {response_data}")
            return jsonify(response_data)
            
        except Exception as api_error:
            print(f"API error: {str(api_error)}")
            return jsonify({'error': 'Failed to analyze image. Please try again.'}), 500
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

# Carbon Tracker Routes
@app.route('/carbon_tracker')
def carbon_tracker():
    return render_template('carbon_tracker.html')

@app.route('/log_activity', methods=['POST'])
def log_activity():
    try:
        data = request.get_json()
        
        # Calculate carbon impact based on activity
        carbon_impact = calculate_carbon_impact(
            data['activityType'],
            data['specificActivity'],
            float(data['quantity']),
            data['unit']
        )
        
        # Store in database
        activity = {
            'user_id': 'default_user',  # Replace with actual user ID when auth is implemented
            'activity_type': data['activityType'],
            'specific_activity': data['specificActivity'],
            'quantity': float(data['quantity']),
            'unit': data['unit'],
            'carbon_impact': carbon_impact,
            'timestamp': datetime.now(IST)
        }
        
        carbon_activities.insert_one(activity)
        
        return jsonify({'success': True, 'carbon_impact': carbon_impact})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_carbon_stats')
def get_carbon_stats():
    try:
        user_id = 'default_user'  # Replace with actual user ID when auth is implemented
        
        # Get current month's activities
        current_month = datetime.now(IST).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_month_activities = list(carbon_activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': current_month}
        }))
        
        # Get last month's activities
        last_month = (current_month - timedelta(days=1)).replace(day=1)
        last_month_activities = list(carbon_activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': last_month, '$lt': current_month}
        }))
        
        # Calculate emissions
        current_emissions = sum(activity['carbon_impact'] for activity in current_month_activities)
        last_emissions = sum(activity['carbon_impact'] for activity in last_month_activities)
        
        # Calculate reduction percentage
        reduction_percentage = 0
        if last_emissions > 0:
            reduction_percentage = ((last_emissions - current_emissions) / last_emissions) * 100
            reduction_percentage = max(0, reduction_percentage)  # Ensure it's not negative
        
        # Get community stats
        total_users = carbon_activities.distinct('user_id')
        active_users = carbon_activities.distinct('user_id', {
            'timestamp': {'$gte': current_month}
        })
        
        # Calculate user rank
        all_user_emissions = []
        for u_id in total_users:
            u_emissions = sum(a['carbon_impact'] for a in carbon_activities.find({
                'user_id': u_id,
                'timestamp': {'$gte': current_month}
            }))
            all_user_emissions.append((u_id, u_emissions))
        
        # Sort by emissions (lower is better)
        all_user_emissions.sort(key=lambda x: x[1])
        user_rank = next(i for i, (u_id, _) in enumerate(all_user_emissions, 1) if u_id == user_id)
        
        return jsonify({
            'personal_emissions': round(current_emissions, 2),
            'reduction_percentage': round(reduction_percentage, 1),
            'community_emissions': round(sum(e for _, e in all_user_emissions), 2),
            'active_users': len(active_users),
            'user_rank': user_rank,
            'total_users': len(total_users)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_leaderboard')
def get_leaderboard():
    try:
        current_month = datetime.now(IST).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get all users and their emissions
        users = carbon_activities.distinct('user_id')
        leaders = []
        
        for user_id in users:
            # Current month emissions
            current_emissions = sum(a['carbon_impact'] for a in carbon_activities.find({
                'user_id': user_id,
                'timestamp': {'$gte': current_month}
            }))
            
            # Last month emissions
            last_month = (current_month - timedelta(days=1)).replace(day=1)
            last_emissions = sum(a['carbon_impact'] for a in carbon_activities.find({
                'user_id': user_id,
                'timestamp': {'$gte': last_month, '$lt': current_month}
            }))
            
            # Calculate reduction
            reduction = 0
            if last_emissions > 0:
                reduction = ((last_emissions - current_emissions) / last_emissions) * 100
                reduction = max(0, reduction)
            
            leaders.append({
                'name': f'User {user_id[-4:]}',  # Using last 4 chars of user_id for demo
                'emissions': round(current_emissions, 2),
                'reduction': round(reduction, 1)
            })
        
        # Sort by emissions (lower is better)
        leaders.sort(key=lambda x: x['emissions'])
        
        return jsonify({'leaders': leaders[:10]})  # Return top 10
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_personalized_tips')
def get_personalized_tips():
    try:
        user_id = 'default_user'  # Replace with actual user ID when auth is implemented
        
        # Get user's recent activities
        recent_activities = list(carbon_activities.find({
            'user_id': user_id
        }).sort('timestamp', -1).limit(10))
        
        # Analyze activities to generate personalized tips
        tips = []
        activity_types = set(a['activity_type'] for a in recent_activities)
        
        if 'transport' in activity_types:
            tips.append({
                'category': 'info',
                'icon': 'fa-car',
                'title': 'Transportation Tip',
                'description': 'Consider carpooling or using public transport to reduce your carbon footprint.'
            })
        
        if 'energy' in activity_types:
            tips.append({
                'category': 'warning',
                'icon': 'fa-bolt',
                'title': 'Energy Saving Tip',
                'description': 'Switch to LED bulbs and turn off appliances when not in use.'
            })
        
        if 'food' in activity_types:
            tips.append({
                'category': 'success',
                'icon': 'fa-utensils',
                'title': 'Food Choice Tip',
                'description': 'Try incorporating more plant-based meals into your diet.'
            })
        
        if 'waste' in activity_types:
            tips.append({
                'category': 'primary',
                'icon': 'fa-recycle',
                'title': 'Waste Management Tip',
                'description': 'Start composting your food waste to reduce methane emissions.'
            })
        
        # Add general tips if needed
        if len(tips) < 3:
            tips.append({
                'category': 'secondary',
                'icon': 'fa-leaf',
                'title': 'General Eco Tip',
                'description': 'Small changes in daily habits can make a big difference in reducing your carbon footprint.'
            })
        
        return jsonify({'tips': tips})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_carbon_impact(activity_type, specific_activity, quantity, unit):
    """Calculate carbon impact in kg CO2e based on activity"""
    # These are simplified calculations for demonstration
    # In a production environment, use more accurate conversion factors
    impact_factors = {
        'transport': {
            'car travel': 0.2,  # kg CO2e per km
            'bus travel': 0.08,
            'train travel': 0.04,
            'air travel': 0.25
        },
        'energy': {
            'electricity usage': 0.5,  # kg CO2e per kWh
            'natural gas': 2.0,  # kg CO2e per cubic meter
            'water usage': 0.001  # kg CO2e per liter
        },
        'food': {
            'meat consumption': 13.3,  # kg CO2e per kg
            'dairy products': 3.2,
            'plant-based meals': 0.5
        },
        'waste': {
            'recycling': -0.5,  # Negative impact (saving)
            'composting': -0.2,
            'landfill waste': 0.7
        }
    }
    
    # Convert units if needed
    if unit == 'miles':
        quantity = quantity * 1.60934  # Convert to kilometers
    
    # Get impact factor
    activity_factors = impact_factors.get(activity_type, {})
    impact_factor = activity_factors.get(specific_activity.lower(), 0.1)  # Default factor if not found
    
    return quantity * impact_factor

# Environmental Impact Dashboard Routes
@app.route('/impact_dashboard')
def impact_dashboard():
    return render_template('impact_dashboard.html')

@app.route('/get_dashboard_data')
def get_dashboard_data():
    try:
        user_id = 'default_user'  # Replace with actual user ID when auth is implemented
        current_time = datetime.now(IST)
        
        # Calculate Eco Score
        eco_score = calculate_eco_score(user_id)
        
        # Get impact metrics
        carbon_stats = calculate_carbon_stats(user_id)
        waste_stats = calculate_waste_stats(user_id)
        
        # Generate timeline
        timeline = generate_impact_timeline(user_id)
        
        # Get AI insights
        insights = generate_ai_insights(user_id)
        
        # Get achievements
        achievements = get_user_achievements(user_id)
        
        # Get active challenges
        challenges = get_active_challenges()
        
        # Get community stats
        community_stats = get_community_stats(user_id)
        
        return jsonify({
            'eco_score': eco_score,
            'carbon_offset': carbon_stats['offset'],
            'waste_reduction': waste_stats['reduction'],
            'progress': {
                'carbonProgress': carbon_stats['progress'],
                'wasteProgress': waste_stats['progress']
            },
            'timeline': timeline,
            'insights': insights,
            'achievements': achievements,
            'challenges': challenges,
            'community': community_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_eco_score(user_id):
    """Calculate user's eco score based on various factors"""
    try:
        # Get recent activities
        recent_carbon = list(mongo.db.carbon_activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': datetime.now(IST) - timedelta(days=30)}
        }))
        
        recent_waste = list(mongo.db.waste_stats.find({
            'user_id': user_id,
            'timestamp': {'$gte': datetime.now(IST) - timedelta(days=30)}
        }))
        
        # Base score
        score = 500
        
        # Add points for carbon reduction
        carbon_impact = sum(activity['carbon_impact'] for activity in recent_carbon)
        if carbon_impact < 100:  # Good carbon impact
            score += 200
        elif carbon_impact < 200:  # Moderate carbon impact
            score += 100
        
        # Add points for waste management
        waste_items = sum(stat['items_recycled'] for stat in recent_waste)
        if waste_items > 50:  # Excellent waste management
            score += 200
        elif waste_items > 25:  # Good waste management
            score += 100
        
        # Add points for consistency
        if len(recent_carbon) > 20 or len(recent_waste) > 20:  # Very active user
            score += 100
        
        return min(1000, score)  # Cap at 1000
    except Exception:
        return 500  # Default score

def calculate_carbon_stats(user_id):
    """Calculate carbon offset and progress"""
    try:
        activities = list(mongo.db.carbon_activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': datetime.now(IST) - timedelta(days=30)}
        }))
        
        total_offset = sum(
            abs(a['carbon_impact']) 
            for a in activities 
            if a['activity_type'] in ['recycling', 'composting']
        )
        
        # Calculate progress (example: 80% progress = offsetting 100kg CO2)
        progress = min(100, (total_offset / 100) * 100)
        
        return {
            'offset': round(total_offset, 2),
            'progress': round(progress, 1)
        }
    except Exception:
        return {'offset': 0, 'progress': 0}

def calculate_waste_stats(user_id):
    """Calculate waste reduction and progress"""
    try:
        stats = list(mongo.db.waste_stats.find({
            'user_id': user_id,
            'timestamp': {'$gte': datetime.now(IST) - timedelta(days=30)}
        }))
        
        total_reduction = sum(stat.get('items_recycled', 0) for stat in stats)
        
        # Calculate progress (example: 80% progress = recycling 50 items)
        progress = min(100, (total_reduction / 50) * 100)
        
        return {
            'reduction': total_reduction,
            'progress': round(progress, 1)
        }
    except Exception:
        return {'reduction': 0, 'progress': 0}

def generate_impact_timeline(user_id):
    """Generate timeline of user's environmental impact"""
    try:
        # Combine carbon and waste activities
        activities = []
        
        # Get carbon activities
        carbon = list(mongo.db.carbon_activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': datetime.now(IST) - timedelta(days=7)}
        }).sort('timestamp', -1))
        
        for activity in carbon:
            activities.append({
                'title': f"{activity['specific_activity'].title()}",
                'description': f"Saved {abs(activity['carbon_impact']):.1f} kg CO₂e",
                'date': activity['timestamp'].strftime('%b %d, %Y')
            })
        
        # Get waste activities
        waste = list(mongo.db.waste_stats.find({
            'user_id': user_id,
            'timestamp': {'$gte': datetime.now(IST) - timedelta(days=7)}
        }).sort('timestamp', -1))
        
        for stat in waste:
            activities.append({
                'title': 'Waste Recycling',
                'description': f"Recycled {stat['items_recycled']} items",
                'date': stat['timestamp'].strftime('%b %d, %Y')
            })
        
        # Sort by date and return most recent 10
        activities.sort(key=lambda x: datetime.strptime(x['date'], '%b %d, %Y'), reverse=True)
        return activities[:10]
    except Exception:
        return []

def generate_ai_insights(user_id):
    """Generate AI-powered insights based on user's activities"""
    try:
        # Get recent activities
        activities = list(mongo.db.carbon_activities.find({
            'user_id': user_id,
            'timestamp': {'$gte': datetime.now(IST) - timedelta(days=30)}
        }))
        
        insights = []
        activity_types = set(a['activity_type'] for a in activities)
        
        # Transport insights
        if 'transport' in activity_types:
            transport_activities = [a for a in activities if a['activity_type'] == 'transport']
            car_usage = sum(a['carbon_impact'] for a in transport_activities if 'car' in a['specific_activity'].lower())
            
            if car_usage > 50:
                insights.append({
                    'icon': 'fa-car',
                    'title': 'Transportation Impact',
                    'description': 'Consider carpooling or using public transport to reduce your carbon footprint by up to 50%.'
                })
        
        # Energy insights
        if 'energy' in activity_types:
            energy_activities = [a for a in activities if a['activity_type'] == 'energy']
            total_energy = sum(a['carbon_impact'] for a in energy_activities)
            
            if total_energy > 100:
                insights.append({
                    'icon': 'fa-bolt',
                    'title': 'Energy Usage',
                    'description': 'Your energy usage is above average. Using LED bulbs and energy-efficient appliances can help reduce consumption.'
                })
        
        # Food insights
        if 'food' in activity_types:
            food_activities = [a for a in activities if a['activity_type'] == 'food']
            meat_consumption = sum(a['carbon_impact'] for a in food_activities if 'meat' in a['specific_activity'].lower())
            
            if meat_consumption > 30:
                insights.append({
                    'icon': 'fa-utensils',
                    'title': 'Dietary Impact',
                    'description': 'Reducing meat consumption by one day per week can save up to 340 kg of CO2 per year.'
                })
        
        # Add general insights if needed
        if len(insights) < 3:
            insights.append({
                'icon': 'fa-leaf',
                'title': 'Eco-friendly Tips',
                'description': 'Small changes like using reusable bags and containers can significantly reduce your environmental impact.'
            })
        
        return insights
    except Exception:
        return []

def get_user_achievements(user_id):
    """Get user's environmental achievements"""
    try:
        # Calculate various metrics
        carbon_saved = sum(a['carbon_impact'] for a in mongo.db.carbon_activities.find({
            'user_id': user_id,
            'activity_type': {'$in': ['recycling', 'composting']}
        }))
        
        items_recycled = sum(s['items_recycled'] for s in mongo.db.waste_stats.find({
            'user_id': user_id
        }))
        
        # Define achievements
        achievements = [
            {
                'icon': 'fa-tree',
                'title': 'Carbon Saver',
                'progress': min(100, (carbon_saved / 100) * 100),
                'unlocked': carbon_saved >= 100
            },
            {
                'icon': 'fa-recycle',
                'title': 'Recycling Pro',
                'progress': min(100, (items_recycled / 50) * 100),
                'unlocked': items_recycled >= 50
            },
            {
                'icon': 'fa-award',
                'title': 'Eco Warrior',
                'progress': min(100, ((carbon_saved + items_recycled) / 200) * 100),
                'unlocked': (carbon_saved + items_recycled) >= 200
            }
        ]
        
        return achievements
    except Exception:
        return []

def get_active_challenges():
    """Get list of active environmental challenges"""
    try:
        current_time = datetime.now(IST)
        
        # Example challenges
        challenges = [
            {
                'title': 'Zero Waste Week',
                'description': 'Minimize your waste production for one week',
                'status': 'Active',
                'active': True,
                'progress': 65,
                'participants': 156
            },
            {
                'title': 'Green Transport',
                'description': 'Use public transport or bike for daily commute',
                'status': '2 days left',
                'active': True,
                'progress': 85,
                'participants': 243
            },
            {
                'title': 'Energy Saver',
                'description': 'Reduce energy consumption by 20%',
                'status': 'Upcoming',
                'active': False,
                'progress': 0,
                'participants': 89
            }
        ]
        
        return challenges
    except Exception:
        return []

def get_community_stats(user_id):
    """Get user's community statistics"""
    try:
        # Calculate user's rank
        all_users = list(mongo.db.carbon_activities.distinct('user_id'))
        user_impacts = []
        
        for u_id in all_users:
            impact = sum(abs(a['carbon_impact']) for a in mongo.db.carbon_activities.find({
                'user_id': u_id,
                'timestamp': {'$gte': datetime.now(IST) - timedelta(days=30)}
            }))
            user_impacts.append((u_id, impact))
        
        # Sort by impact (higher is better)
        user_impacts.sort(key=lambda x: x[1], reverse=True)
        rank = next(i for i, (u_id, _) in enumerate(user_impacts, 1) if u_id == user_id)
        
        # Calculate percentiles
        total_users = len(user_impacts)
        rank_percentile = ((total_users - rank) / total_users) * 100
        
        # Calculate contribution
        total_impact = sum(impact for _, impact in user_impacts)
        user_impact = next(impact for u_id, impact in user_impacts if u_id == user_id)
        contribution = (user_impact / total_impact * 100) if total_impact > 0 else 0
        
        return {
            'rank': rank,
            'rank_percentile': round(rank_percentile, 1),
            'contribution': round(contribution, 1)
        }
    except Exception:
        return {'rank': 0, 'rank_percentile': 0, 'contribution': 0}

if __name__ == '__main__':
    app.run(debug=True, port=5676)
