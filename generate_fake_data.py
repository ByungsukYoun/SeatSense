"""
Library Seat Detection - Fake Data Generator
Generates one month of simulated data (for analytics testing)
"""

import sqlite3
import random
from datetime import datetime, timedelta

DATABASE_FILE = 'seat_data.db'

# Camera and seat configuration
CAMERAS = {
    'cam1': ['1', '2', '3', '4', '5', '6', '7', '8'],  # 8 seats
    'cam2': ['1', '2', '3', '4', '5', '6', '7']        # 7 seats
}

def get_occupancy_probability(hour, day_of_week):
    """
    Calculate occupancy probability based on hour and day of week
    Simulates real library usage patterns
    """
    
    # Weekends are less busy
    weekend_factor = 0.6 if day_of_week >= 5 else 1.0
    
    # Hourly pattern (0-23)
    hourly_pattern = {
        0: 0.05, 1: 0.02, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.02,
        6: 0.05, 7: 0.15, 8: 0.35, 9: 0.55, 10: 0.70, 11: 0.75,
        12: 0.60,  # Lunch hour - slight decrease
        13: 0.65, 14: 0.80, 15: 0.85, 16: 0.80, 17: 0.70,
        18: 0.55,  # Evening hours
        19: 0.65, 20: 0.70, 21: 0.60, 22: 0.40, 23: 0.15
    }
    
    base_probability = hourly_pattern.get(hour, 0.3)
    
    # Add slight randomness
    noise = random.uniform(-0.1, 0.1)
    
    probability = base_probability * weekend_factor + noise
    return max(0.0, min(1.0, probability))

def generate_fake_data(days=30):
    """
    Generate N days of fake data
    
    Args:
        days: Number of days to generate (default 30)
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS occupancy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_id TEXT NOT NULL,
            seat_id TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    
    # Delete existing data (optional)
    cursor.execute('DELETE FROM occupancy_log')
    print("Existing data cleared")
    
    # Start date (N days ago)
    start_date = datetime.now() - timedelta(days=days)
    
    total_records = 0
    
    print(f"\nStarting data generation for {days} days...")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} ~ {datetime.now().strftime('%Y-%m-%d')}")
    
    # Generate data for each day and hour
    for day in range(days + 1):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        
        # Operating hours: 6am ~ 11pm
        for hour in range(6, 24):
            # 4 records per hour (every 15 minutes)
            for minute in [0, 15, 30, 45]:
                timestamp = current_date.replace(hour=hour, minute=minute, second=0)
                
                # Skip if timestamp is in the future
                if timestamp > datetime.now():
                    continue
                
                occupancy_prob = get_occupancy_probability(hour, day_of_week)
                
                # For each camera and each seat
                for camera_id, seats in CAMERAS.items():
                    for seat_id in seats:
                        # Slight popularity difference per seat
                        seat_factor = 1.0 + (int(seat_id) % 3) * 0.1  # Seats 1-3 are slightly more popular
                        adjusted_prob = min(1.0, occupancy_prob * seat_factor)
                        
                        # Determine status
                        status = 'Occupied' if random.random() < adjusted_prob else 'Available'
                        
                        cursor.execute('''
                            INSERT INTO occupancy_log (timestamp, camera_id, seat_id, status)
                            VALUES (?, ?, ?, ?)
                        ''', (timestamp.strftime('%Y-%m-%d %H:%M:%S'), camera_id, seat_id, status))
                        
                        total_records += 1
        
        # Show progress
        if day % 5 == 0:
            print(f" {current_date.strftime('%Y-%m-%d')} done ({day+1}/{days+1} days)")
    
    conn.commit()
    conn.close()
    
    print(f"\nData generation complete!")
    print(f"Total records generated: {total_records:,}")
    print(f"Saved to: {DATABASE_FILE}")
    
    return total_records

def show_sample_data():
    """Display sample of generated data"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Last 10 records
    cursor.execute('''
        SELECT timestamp, camera_id, seat_id, status 
        FROM occupancy_log 
        ORDER BY timestamp DESC 
        LIMIT 10
    ''')
    
    print("\nLast 10 records:")
    print("-" * 60)
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1]} | Seat {row[2]} | {row[3]}")
    
    # Statistics
    cursor.execute('SELECT COUNT(*) FROM occupancy_log')
    total = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT status, COUNT(*) as count 
        FROM occupancy_log 
        GROUP BY status
    ''')
    
    print("\nOverall statistics:")
    print("-" * 60)
    print(f"  Total records: {total:,}")
    for row in cursor.fetchall():
        percentage = row[1] / total * 100
        print(f"  {row[0]}: {row[1]:,} ({percentage:.1f}%)")
    
    conn.close()

if __name__ == '__main__':
    print("="*60)
    print("     Library Seat Detection - Fake Data Generator")
    print("="*60)
    
    # Generate 30 days of data
    generate_fake_data(days=30)
    
    # Show sample
    show_sample_data()