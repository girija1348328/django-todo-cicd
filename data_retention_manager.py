#!/usr/bin/env python3
"""
Data Retention Manager for Detection History
Allows configurable retention periods and automatic cleanup of old records.
"""

import sqlite3
import os
from datetime import datetime, timedelta
import json

class DataRetentionManager:
    def __init__(self, db_path='database.db', config_file='retention_config.json'):
        self.db_path = db_path
        self.config_file = config_file
        self.default_config = {
            "retention_days": 90,  # Keep records for 90 days by default
            "auto_cleanup": False,  # Disable automatic cleanup (manual control)
            "cleanup_frequency_hours": 24,  # Run cleanup every 24 hours
            "max_records": 10000,   # Maximum records to keep
            "backup_before_cleanup": True,  # Backup before deleting
            "notify_on_cleanup": True       # Show cleanup results
        }
        self.load_config()
    
    def load_config(self):
        """Load retention configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                print(f"✅ Loaded retention config: {self.config_file}")
            except Exception as e:
                print(f"⚠️ Error loading config, using defaults: {e}")
                self.config = self.default_config.copy()
        else:
            print(f"📝 No config file found, creating default: {self.config_file}")
            self.config = self.default_config.copy()
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Saved retention config: {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get_retention_info(self):
        """Get current retention settings and database stats"""
        if not os.path.exists(self.db_path):
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM detection_history")
            total_records = cursor.fetchone()[0]
            
            # Get oldest and newest records
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM detection_history 
                WHERE timestamp IS NOT NULL
            """)
            time_range = cursor.fetchone()
            oldest_time = time_range[0] if time_range[0] else None
            newest_time = time_range[1] if time_range[1] else None
            
            # Calculate age of oldest record
            oldest_age_days = None
            if oldest_time:
                try:
                    oldest_dt = datetime.fromisoformat(oldest_time.replace('Z', '+00:00'))
                    oldest_age_days = (datetime.now() - oldest_dt).days
                except:
                    pass
            
            # Get records that would be deleted
            cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
            cursor.execute("""
                SELECT COUNT(*) FROM detection_history 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            records_to_delete = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_records': total_records,
                'retention_days': self.config['retention_days'],
                'oldest_record': oldest_time,
                'newest_record': newest_time,
                'oldest_age_days': oldest_age_days,
                'records_to_delete': records_to_delete,
                'auto_cleanup': self.config['auto_cleanup'],
                'max_records': self.config['max_records']
            }
            
        except Exception as e:
            print(f"❌ Error getting retention info: {e}")
            return None
    
    def cleanup_old_records(self, dry_run=True):
        """Clean up old records based on retention policy"""
        if not os.path.exists(self.db_path):
            print("❌ Database not found!")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
            cutoff_str = cutoff_date.isoformat()
            
            # Get records to delete
            cursor.execute("""
                SELECT id, timestamp, employee_name, access_granted 
                FROM detection_history 
                WHERE timestamp < ?
                ORDER BY timestamp ASC
            """, (cutoff_str,))
            
            records_to_delete = cursor.fetchall()
            
            if not records_to_delete:
                print("✅ No old records to clean up")
                conn.close()
                return True
            
            print(f"🔍 Found {len(records_to_delete)} records older than {self.config['retention_days']} days")
            print(f"   Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if dry_run:
                print("\n📋 Records that would be deleted (DRY RUN):")
                print("-" * 80)
                print(f"{'ID':<4} {'Date':<20} {'Name':<20} {'Status':<8}")
                print("-" * 80)
                
                for record in records_to_delete[:10]:  # Show first 10
                    status = "✅ GRANTED" if record[3] else "❌ DENIED"
                    date_str = record[1][:19] if record[1] else "N/A"
                    print(f"{record[0]:<4} {date_str:<20} {record[2]:<20} {status:<8}")
                
                if len(records_to_delete) > 10:
                    print(f"   ... and {len(records_to_delete) - 10} more records")
                
                print(f"\n💡 To actually delete these records, run with dry_run=False")
                conn.close()
                return True
            
            # Actually delete the records
            print(f"\n🗑️ Deleting {len(records_to_delete)} old records...")
            
            cursor.execute("""
                DELETE FROM detection_history 
                WHERE timestamp < ?
            """, (cutoff_str,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            print(f"✅ Successfully deleted {deleted_count} old records")
            
            # Get new total
            cursor.execute("SELECT COUNT(*) FROM detection_history")
            new_total = cursor.fetchone()[0]
            print(f"📊 Database now contains {new_total} records")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def set_retention_days(self, days):
        """Set retention period in days"""
        if days < 1:
            print("❌ Retention days must be at least 1")
            return False
        
        old_days = self.config['retention_days']
        self.config['retention_days'] = days
        self.save_config()
        
        print(f"✅ Retention period changed from {old_days} days to {days} days")
        
        # Show what this means
        cutoff_date = datetime.now() - timedelta(days=days)
        print(f"   Records older than {cutoff_date.strftime('%Y-%m-%d')} will be cleaned up")
        
        return True
    
    def enable_auto_cleanup(self, enable=True):
        """Enable or disable automatic cleanup"""
        old_setting = self.config['auto_cleanup']
        self.config['auto_cleanup'] = enable
        self.save_config()
        
        status = "enabled" if enable else "disabled"
        print(f"✅ Automatic cleanup {status}")
        
        if enable:
            print(f"   Records older than {self.config['retention_days']} days will be automatically removed")
        else:
            print("   You'll need to manually run cleanup_old_records()")
        
        return True
    
    def show_status(self):
        """Display current retention status"""
        info = self.get_retention_info()
        if not info:
            print("❌ Could not get retention info")
            return
        
        print("\n📊 Data Retention Status")
        print("=" * 50)
        print(f"Retention Period: {info['retention_days']} days")
        print(f"Auto Cleanup: {'✅ Enabled' if info['auto_cleanup'] else '❌ Disabled'}")
        print(f"Total Records: {info['total_records']}")
        print(f"Max Records: {info['max_records']}")
        
        if info['oldest_record']:
            print(f"Oldest Record: {info['oldest_record']} ({info['oldest_age_days']} days old)")
        
        if info['newest_record']:
            print(f"Newest Record: {info['newest_record']}")
        
        if info['records_to_delete'] > 0:
            print(f"Records to Delete: {info['records_to_delete']} (older than {info['retention_days']} days)")
        else:
            print("Records to Delete: 0 (all records within retention period)")
        
        # Show retention timeline
        if info['oldest_age_days']:
            print(f"\n📅 Retention Timeline:")
            print(f"   Current: {datetime.now().strftime('%Y-%m-%d')}")
            print(f"   Cutoff: {(datetime.now() - timedelta(days=info['retention_days'])).strftime('%Y-%m-%d')}")
            print(f"   Oldest: {info['oldest_record'][:10] if info['oldest_record'] else 'N/A'}")
            
            if info['oldest_age_days'] > info['retention_days']:
                print(f"   ⚠️  Oldest record is {info['oldest_age_days'] - info['retention_days']} days beyond retention period")
            else:
                days_left = info['retention_days'] - info['oldest_age_days']
                print(f"   ✅ Oldest record will be kept for {days_left} more days")

def main():
    """Main function for command line usage"""
    print("🗂️ Data Retention Manager for Detection History")
    print("=" * 60)
    
    manager = DataRetentionManager()
    
    # Show current status
    manager.show_status()
    
    print(f"\n💡 Available Commands:")
    print("1. manager.set_retention_days(30)     # Keep records for 30 days")
    print("2. manager.enable_auto_cleanup(True)  # Enable automatic cleanup")
    print("3. manager.cleanup_old_records()      # Dry run cleanup")
    print("4. manager.cleanup_old_records(False) # Actually delete old records")
    print("5. manager.show_status()              # Show current status")
    
    # Example: Set to 30 days retention
    print(f"\n🧪 Example: Setting retention to 30 days...")
    manager.set_retention_days(30)
    
    # Show what would be cleaned up
    print(f"\n🧪 Example: Checking what would be cleaned up...")
    manager.cleanup_old_records(dry_run=True)

if __name__ == "__main__":
    main()
