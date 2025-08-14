# 🗂️ Data Retention Management for Detection History

## 📋 Overview

Your detection history system now supports **configurable data retention periods** instead of storing data forever. This helps manage database size and comply with data privacy requirements.

## ⚙️ Current Default Settings

- **Retention Period**: 90 days
- **Auto Cleanup**: Enabled
- **Max Records**: 10,000
- **Cleanup Frequency**: Every 24 hours

## 🚀 Quick Start

### 1. Check Current Status
```bash
python data_retention_manager.py
```

### 2. Interactive Management
```bash
python manage_retention.py
```

### 3. Set Retention to 30 Days
```bash
python -c "
from data_retention_manager import DataRetentionManager
manager = DataRetentionManager()
manager.set_retention_days(30)
manager.show_status()
"
```

## 📊 Retention Periods

| Period | Use Case | Records Kept |
|--------|----------|--------------|
| **7 days** | High-security areas | 1 week of history |
| **30 days** | Standard security | 1 month of history |
| **90 days** | Compliance | 3 months of history |
| **365 days** | Long-term audit | 1 year of history |
| **Forever** | Legal requirements | All records (manual cleanup) |

## 🔧 Configuration Options

### retention_config.json
```json
{
  "retention_days": 90,           // Keep records for 90 days
  "auto_cleanup": true,           // Enable automatic cleanup
  "cleanup_frequency_hours": 24,  // Run cleanup every 24 hours
  "max_records": 10000,           // Maximum records to keep
  "backup_before_cleanup": true,  // Backup before deleting
  "notify_on_cleanup": true       // Show cleanup results
}
```

## 🎯 Common Use Cases

### High-Security Environment (7 days)
```bash
python -c "
from data_retention_manager import DataRetentionManager
manager = DataRetentionManager()
manager.set_retention_days(7)
manager.enable_auto_cleanup(True)
print('✅ Set to 7-day retention with auto-cleanup')
"
```

### Compliance Environment (90 days)
```bash
python -c "
from data_retention_manager import DataRetentionManager
manager = DataRetentionManager()
manager.set_retention_days(90)
manager.enable_auto_cleanup(True)
print('✅ Set to 90-day retention with auto-cleanup')
"
```

### Legal Requirements (Keep forever)
```bash
python -c "
from data_retention_manager import DataRetentionManager
manager = DataRetentionManager()
manager.enable_auto_cleanup(False)
print('✅ Disabled auto-cleanup - records kept forever')
"
```

## 🧹 Manual Cleanup

### Check What Would Be Deleted
```bash
python -c "
from data_retention_manager import DataRetentionManager
manager = DataRetentionManager()
manager.cleanup_old_records(dry_run=True)
"
```

### Actually Delete Old Records
```bash
python -c "
from data_retention_manager import DataRetentionManager
manager = DataRetentionManager()
manager.cleanup_old_records(dry_run=False)
"
```

## 📈 Database Size Management

### Current Database Status
```bash
python check_db_fixed.py
```

### Monitor Growth
```bash
python -c "
from data_retention_manager import DataRetentionManager
manager = DataRetentionManager()
info = manager.get_retention_info()
print(f'Total Records: {info[\"total_records\"]}')
print(f'Records to Delete: {info[\"records_to_delete\"]}')
print(f'Oldest Record Age: {info[\"oldest_age_days\"]} days')
"
```

## 🔒 Security Considerations

### Before Cleanup
- **Backup**: Always backup before major cleanup
- **Audit**: Review what will be deleted
- **Compliance**: Ensure retention meets legal requirements

### After Cleanup
- **Verify**: Check that cleanup was successful
- **Monitor**: Watch for any issues
- **Document**: Keep records of cleanup operations

## 🚨 Troubleshooting

### Records Not Being Deleted
1. Check if auto-cleanup is enabled
2. Verify retention period is set correctly
3. Check database permissions
4. Review error logs

### Database Growing Too Fast
1. Reduce retention period
2. Enable auto-cleanup
3. Set lower max_records limit
4. Monitor detection frequency

### Configuration Not Saving
1. Check file permissions
2. Verify JSON syntax
3. Restart the application
4. Check disk space

## 📱 Integration with Flask App

### Automatic Cleanup
The system can be integrated with your Flask app to run cleanup automatically:

```python
# In your Flask app
from data_retention_manager import DataRetentionManager

@app.before_first_request
def setup_retention():
    manager = DataRetentionManager()
    if manager.config['auto_cleanup']:
        # Run cleanup if needed
        manager.cleanup_old_records(dry_run=False)
```

### API Endpoint for Cleanup
```python
@app.route('/api/cleanup', methods=['POST'])
def cleanup_old_records():
    manager = DataRetentionManager()
    success = manager.cleanup_old_records(dry_run=False)
    return {'success': success, 'message': 'Cleanup completed'}
```

## 💡 Best Practices

1. **Start Conservative**: Begin with 90 days, adjust as needed
2. **Monitor Growth**: Check database size regularly
3. **Test Cleanup**: Always run dry-run first
4. **Backup Regularly**: Keep backups before major operations
5. **Document Changes**: Record retention policy changes
6. **Compliance Check**: Ensure retention meets legal requirements

## 🔄 Migration from Forever Storage

If you want to migrate from storing forever to a retention period:

1. **Set Retention Period**:
   ```bash
   python manage_retention.py
   # Choose option 2, enter desired days
   ```

2. **Enable Auto-Cleanup**:
   ```bash
   python manage_retention.py
   # Choose option 3, enter 'y'
   ```

3. **Test Cleanup**:
   ```bash
   python manage_retention.py
   # Choose option 4 (dry run)
   ```

4. **Run Cleanup**:
   ```bash
   python manage_retention.py
   # Choose option 5, confirm deletion
   ```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python check_db_fixed.py` for database status
3. Review error messages in the console
4. Check file permissions and disk space

---

**Remember**: Data retention is important for both security and compliance. Choose a retention period that balances your needs for historical data with storage and privacy requirements.
