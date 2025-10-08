#!/usr/bin/env python3
"""
Performance comparison script to demonstrate HDD optimization benefits.
Compares batch insert vs. individual insert performance.
"""

import time
import pandas as pd
import sqlite3
import os
import tempfile
from datetime import datetime, timedelta

def test_individual_inserts(db_path, test_data):
    """Test performance with individual INSERT statements."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            UNIQUE(symbol, date)
        )
    ''')
    
    start_time = time.time()
    inserted = 0
    
    for idx, row in test_data.iterrows():
        try:
            cursor.execute(
                """
                INSERT INTO stocks (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ('TEST.BSE', idx.strftime('%Y-%m-%d'), row['open'], 
                 row['high'], row['low'], row['close'], int(row['volume']))
            )
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    
    conn.commit()
    conn.close()
    
    elapsed = time.time() - start_time
    return elapsed, inserted

def test_batch_inserts(db_path, test_data):
    """Test performance with batch INSERT using executemany."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable optimizations
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA cache_size=10000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            UNIQUE(symbol, date)
        )
    ''')
    
    # Create index
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stocks_symbol_date 
        ON stocks(symbol, date DESC)
    ''')
    
    # Prepare batch data
    rows_to_insert = []
    for idx, row in test_data.iterrows():
        rows_to_insert.append((
            'TEST.BSE', idx.strftime('%Y-%m-%d'), row['open'],
            row['high'], row['low'], row['close'], int(row['volume'])
        ))
    
    start_time = time.time()
    
    cursor.execute("BEGIN TRANSACTION")
    cursor.executemany(
        """
        INSERT OR IGNORE INTO stocks (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert
    )
    inserted = cursor.rowcount
    conn.commit()
    conn.close()
    
    elapsed = time.time() - start_time
    return elapsed, inserted

def main():
    print("=" * 70)
    print("HDD OPTIMIZATION PERFORMANCE COMPARISON")
    print("=" * 70)
    print()
    
    # Create test data
    print("Generating test data...")
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    test_data = pd.DataFrame({
        'open': [100 + i*0.5 for i in range(1000)],
        'high': [105 + i*0.5 for i in range(1000)],
        'low': [95 + i*0.5 for i in range(1000)],
        'close': [102 + i*0.5 for i in range(1000)],
        'volume': [1000000 + i*1000 for i in range(1000)]
    }, index=dates)
    print(f"Generated {len(test_data)} test records")
    print()
    
    # Test 1: Individual inserts (without optimizations)
    print("Test 1: Individual INSERT statements (no optimizations)")
    print("-" * 70)
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tf:
        db_path1 = tf.name
    
    try:
        time1, count1 = test_individual_inserts(db_path1, test_data)
        print(f"  Records inserted: {count1}")
        print(f"  Time taken: {time1:.3f} seconds")
        print(f"  Throughput: {count1/time1:.1f} records/second")
    finally:
        if os.path.exists(db_path1):
            os.unlink(db_path1)
    
    print()
    
    # Test 2: Batch inserts with optimizations
    print("Test 2: Batch INSERT with HDD optimizations (WAL, cache, indexing)")
    print("-" * 70)
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tf:
        db_path2 = tf.name
    
    try:
        time2, count2 = test_batch_inserts(db_path2, test_data)
        print(f"  Records inserted: {count2}")
        print(f"  Time taken: {time2:.3f} seconds")
        print(f"  Throughput: {count2/time2:.1f} records/second")
    finally:
        if os.path.exists(db_path2):
            os.unlink(db_path2)
            # Clean up WAL files
            for suffix in ['-wal', '-shm']:
                wal_file = db_path2 + suffix
                if os.path.exists(wal_file):
                    os.unlink(wal_file)
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Individual inserts: {time1:.3f}s ({count1/time1:.1f} records/sec)")
    print(f"  Batch + HDD optimizations: {time2:.3f}s ({count2/time2:.1f} records/sec)")
    print()
    speedup = time1 / time2
    print(f"  🚀 SPEEDUP: {speedup:.1f}x faster with HDD optimizations!")
    print(f"  ⏱️  TIME SAVED: {time1 - time2:.3f} seconds ({(time1-time2)/time1*100:.1f}% reduction)")
    print("=" * 70)

if __name__ == "__main__":
    main()
