"""
Library Seat Detection - Database Module
Seat status logging and analysis using SQLite
"""

import sqlite3
from datetime import datetime, timedelta
import os

DATABASE_FILE = 'seat_data.db'

def get_connection():
    """Connect to the database"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row  # Allows dictionary-style access
    return conn

def init_database():
    """Initialize database - create tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Seat status log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS occupancy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_id TEXT NOT NULL,
            seat_id TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    
    # Hourly statistics table (for aggregation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hourly_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            hour INTEGER NOT NULL,
            camera_id TEXT NOT NULL,
            total_seats INTEGER,
            avg_occupied REAL,
            avg_available REAL,
            UNIQUE(date, hour, camera_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def log_occupancy(camera_id, seat_statuses):
    """
    Log current seat status
    
    Args:
        camera_id: Camera ID (cam1, cam2)
        seat_statuses: {seat_id: status} dictionary
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for seat_id, status in seat_statuses.items():
        cursor.execute('''
            INSERT INTO occupancy_log (timestamp, camera_id, seat_id, status)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, camera_id, seat_id, status))
    
    conn.commit()
    conn.close()

def get_hourly_stats(date=None):
    """
    Retrieve hourly statistics
    
    Args:
        date: Date to query (default: today)
    
    Returns:
        List of occupancy rates by hour
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as total_records,
            SUM(CASE WHEN status = 'Occupied' THEN 1 ELSE 0 END) as occupied_count,
            SUM(CASE WHEN status = 'Available' THEN 1 ELSE 0 END) as available_count
        FROM occupancy_log
        WHERE date(timestamp) = ?
        GROUP BY hour
        ORDER BY hour
    ''', (date,))
    
    results = []
    for row in cursor.fetchall():
        total = row['occupied_count'] + row['available_count']
        occupancy_rate = (row['occupied_count'] / total * 100) if total > 0 else 0
        results.append({
            'hour': int(row['hour']),
            'occupancy_rate': round(occupancy_rate, 1),
            'occupied': row['occupied_count'],
            'available': row['available_count']
        })
    
    conn.close()
    return results

def get_peak_hours(days=7):
    """
    Peak hour analysis (last N days)
    
    Returns:
        Top 5 busiest hours
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as total_records,
            SUM(CASE WHEN status = 'Occupied' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as occupancy_rate
        FROM occupancy_log
        WHERE timestamp >= datetime('now', ?)
        GROUP BY hour
        ORDER BY occupancy_rate DESC
        LIMIT 5
    ''', (f'-{days} days',))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'hour': int(row['hour']),
            'hour_display': f"{int(row['hour']):02d}:00",
            'occupancy_rate': round(row['occupancy_rate'], 1)
        })
    
    conn.close()
    return results

def get_seat_popularity(days=7):
    """
    Seat popularity analysis
    
    Returns:
        Seat usage rates (sorted by most popular)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            camera_id,
            seat_id,
            COUNT(*) as total_records,
            SUM(CASE WHEN status = 'Occupied' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as usage_rate
        FROM occupancy_log
        WHERE timestamp >= datetime('now', ?)
        GROUP BY camera_id, seat_id
        ORDER BY usage_rate DESC
    ''', (f'-{days} days',))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'camera_id': row['camera_id'],
            'seat_id': row['seat_id'],
            'usage_rate': round(row['usage_rate'], 1)
        })
    
    conn.close()
    return results

def get_daily_summary(days=7):
    """
    Daily summary statistics
    
    Returns:
        Average occupancy rate per day for the last N days
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            date(timestamp) as date,
            COUNT(*) as total_records,
            SUM(CASE WHEN status = 'Occupied' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as occupancy_rate
        FROM occupancy_log
        WHERE timestamp >= datetime('now', ?)
        GROUP BY date(timestamp)
        ORDER BY date DESC
    ''', (f'-{days} days',))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'date': row['date'],
            'occupancy_rate': round(row['occupancy_rate'], 1)
        })
    
    conn.close()
    return results

def get_current_vs_average():
    """
    Compare current occupancy rate vs average occupancy rate
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Average occupancy rate for the current hour (last 7 days)
    current_hour = datetime.now().hour
    
    cursor.execute('''
        SELECT 
            AVG(CASE WHEN status = 'Occupied' THEN 1.0 ELSE 0.0 END) * 100 as avg_occupancy
        FROM occupancy_log
        WHERE strftime('%H', timestamp) = ?
        AND timestamp >= datetime('now', '-7 days')
    ''', (f'{current_hour:02d}',))
    
    row = cursor.fetchone()
    avg_occupancy = round(row['avg_occupancy'], 1) if row['avg_occupancy'] else 0
    
    conn.close()
    return {
        'current_hour': current_hour,
        'average_occupancy_this_hour': avg_occupancy
    }

def get_total_records():
    """Retrieve total number of records"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as count FROM occupancy_log')
    row = cursor.fetchone()
    
    conn.close()
    return row['count']

def cleanup_old_data(days=30):
    """Clean up old data (default: older than 30 days)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        DELETE FROM occupancy_log
        WHERE timestamp < datetime('now', ?)
    ''', (f'-{days} days',))
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted

# For testing
if __name__ == '__main__':
    init_database()
    print(f"Database file: {DATABASE_FILE}")
    print(f"Total records: {get_total_records()}")