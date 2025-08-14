# Detection History Database System

This system automatically stores all face detection and recognition events in a local database, providing persistent storage and advanced querying capabilities for the IG Security System.

## 🚀 Features

- **Automatic Storage**: Every detection event is automatically stored in the database
- **Rich Data**: Stores employee details, camera information, confidence scores, and timestamps
- **Real-time Updates**: Live detection cards now persist to database
- **Advanced Queries**: Filter by access type, camera, date range, and more
- **Statistics**: Get insights into detection patterns and camera usage
- **History Management**: View, filter, and manage detection records

## 📋 Database Schema

The `detection_history` table stores the following information:

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `timestamp` | DateTime | When the detection occurred |
| `access_granted` | Boolean | True for granted, False for denied |
| `employee_name` | String(100) | Name of detected person |
| `designation` | String(100) | Job title/role |
| `department` | String(100) | Department/division |
| `confidence_score` | Float | Detection confidence (0.0-1.0) |
| `location` | String(100) | Camera location/name |
| `camera_name` | String(100) | Camera identifier |
| `camera_id` | String(50) | Camera system ID |
| `detection_type` | String(50) | Type of detection |
| `image_data` | Text | Optional base64 image data |
| `employee_id` | Integer | Foreign key to employee table |

## 🛠️ Setup Instructions

### 1. Create the Database Table

Run the setup script to create the new table:

```bash
python create_detection_history_table.py
```

This will:
- Create the `detection_history` table
- Verify the table structure
- Show confirmation of successful setup

### 2. Restart Your Flask Application

After creating the table, restart your Flask application to load the new routes and models.

### 3. Test the API Endpoints

Use the test script to verify everything is working:

```bash
python test_detection_history_api.py
```

## 🔌 API Endpoints

### Store Detection Event
```
POST /api/detection_history
```
Stores a new detection event. Called automatically by the frontend.

**Request Body:**
```json
{
    "access": false,
    "employee_name": "John Doe",
    "designation": "Engineer",
    "department": "IT",
    "confidence": 0.95,
    "location": "Main Entrance",
    "camera_name": "Cam01",
    "camera_id": "cam_001",
    "detection_type": "face_detection"
}
```

### Get Detection History
```
GET /api/detection_history
```
Retrieves detection history with optional filtering.

**Query Parameters:**
- `limit`: Number of records to return (default: 100)
- `offset`: Number of records to skip (default: 0)
- `access`: Filter by access type (`granted` or `denied`)
- `camera`: Filter by camera name
- `date_from`: Filter from date (ISO format)
- `date_to`: Filter to date (ISO format)

**Example:**
```
GET /api/detection_history?access=denied&limit=50&camera=Cam01
```

### Get Statistics
```
GET /api/detection_history/stats
```
Returns detection statistics for analysis.

**Query Parameters:**
- `days`: Number of days to analyze (default: 7)

**Response:**
```json
{
    "success": true,
    "stats": {
        "total_detections": 150,
        "granted_count": 120,
        "denied_count": 30,
        "granted_percentage": 80.0,
        "denied_percentage": 20.0,
        "camera_breakdown": {
            "Cam01": 80,
            "Cam02": 70
        },
        "date_range_days": 7
    }
}
```

### Delete Specific Record
```
DELETE /api/detection_history/<id>
```
Deletes a specific detection record by ID.

### Clear All History
```
POST /api/detection_history/clear?confirm=true
```
⚠️ **Dangerous Operation**: Clears all detection history. Requires confirmation.

## 🎯 Frontend Integration

The frontend automatically integrates with the database:

1. **Live Detection Cards**: Each detection event is stored in the database
2. **History Sidebar**: Loads data from the database instead of just local memory
3. **Persistent Storage**: Detection history survives page refreshes and application restarts
4. **Real-time Updates**: New detections appear in both live cards and history

## 📊 Usage Examples

### View Recent Denied Access
```bash
curl "http://localhost:5000/api/detection_history?access=denied&limit=10"
```

### Get Weekly Statistics
```bash
curl "http://localhost:5000/api/detection_history/stats?days=7"
```

### Filter by Camera
```bash
curl "http://localhost:5000/api/detection_history?camera=Cam01&limit=20"
```

### Date Range Filter
```bash
curl "http://localhost:5000/api/detection_history?date_from=2024-01-01&date_to=2024-01-31"
```

## 🔍 Database Queries

You can also query the database directly using SQL:

```sql
-- Get all denied access attempts today
SELECT * FROM detection_history 
WHERE access_granted = 0 
AND DATE(timestamp) = CURDATE();

-- Get detection count by camera
SELECT camera_name, COUNT(*) as count 
FROM detection_history 
GROUP BY camera_name;

-- Get employee access patterns
SELECT employee_name, 
       COUNT(*) as total_detections,
       SUM(CASE WHEN access_granted = 1 THEN 1 ELSE 0 END) as granted,
       SUM(CASE WHEN access_granted = 0 THEN 1 ELSE 0 END) as denied
FROM detection_history 
WHERE employee_name != 'Unknown Visitor'
GROUP BY employee_name;
```

## 🚨 Troubleshooting

### Common Issues

1. **Table Not Found**: Run `create_detection_history_table.py` again
2. **API 404 Errors**: Ensure the Flask app has been restarted after setup
3. **Database Connection Issues**: Check your database configuration in `config.py`
4. **Permission Errors**: Verify database user has CREATE/INSERT permissions

### Debug Mode

Enable debug logging in the detection history routes by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Check Database Status

Use the test script to verify API connectivity:
```bash
python test_detection_history_api.py
```

## 🔮 Future Enhancements

- **Image Storage**: Store actual detection images in the database
- **Advanced Analytics**: Machine learning insights on detection patterns
- **Export Features**: CSV/Excel export of detection history
- **Real-time Dashboard**: Live statistics and monitoring
- **Alert System**: Notifications for unusual detection patterns

## 📝 Notes

- The system automatically handles both known employees and unknown visitors
- Detection events are stored with UTC timestamps
- The database uses SQLAlchemy ORM for easy querying
- All API endpoints return JSON responses with consistent error handling
- The frontend automatically retries failed database operations

## 🤝 Support

If you encounter issues:
1. Check the console logs for error messages
2. Verify the database table exists and has correct structure
3. Test the API endpoints individually
4. Check Flask application logs for backend errors

---

**Happy Detecting! 🎯**
