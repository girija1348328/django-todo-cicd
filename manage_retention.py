#!/usr/bin/env python3
"""
Simple script to manage detection history retention settings.
Easy to use for setting retention periods and cleaning up old data.
"""

from data_retention_manager import DataRetentionManager

def main():
    print("🗂️ Detection History Retention Manager")
    print("=" * 50)
    
    manager = DataRetentionManager()
    
    while True:
        print("\n📋 Available Options:")
        print("1. Show current retention status")
        print("2. Set retention period (days)")
        print("3. Enable/disable auto cleanup")
        print("4. Check what would be deleted (dry run)")
        print("5. Clean up old records")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            manager.show_status()
            
        elif choice == '2':
            try:
                days = int(input("Enter retention period in days (e.g., 30, 90, 365): "))
                if days > 0:
                    manager.set_retention_days(days)
                else:
                    print("❌ Days must be greater than 0")
            except ValueError:
                print("❌ Please enter a valid number")
                
        elif choice == '3':
            enable = input("Enable auto cleanup? (y/n): ").lower().strip()
            if enable in ['y', 'yes']:
                manager.enable_auto_cleanup(True)
            elif enable in ['n', 'no']:
                manager.enable_auto_cleanup(False)
            else:
                print("❌ Please enter y or n")
                
        elif choice == '4':
            print("\n🔍 Checking what would be deleted...")
            manager.cleanup_old_records(dry_run=True)
            
        elif choice == '5':
            confirm = input("⚠️ This will permanently delete old records. Continue? (y/n): ").lower().strip()
            if confirm in ['y', 'yes']:
                print("\n🗑️ Cleaning up old records...")
                manager.cleanup_old_records(dry_run=False)
            else:
                print("❌ Cleanup cancelled")
                
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6")

if __name__ == "__main__":
    main()
