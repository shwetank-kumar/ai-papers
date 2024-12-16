import sqlite3
from datetime import datetime

# Connect to your SQLite database
conn = sqlite3.connect('paper_summaries.db')
cursor = conn.cursor()

# Define the date range
start_date = '01-11-2024'  # dd-mm-yyyy
end_date = '08-11-2024'    # dd-mm-yyyy

# Convert dates to datetime objects for comparison
start_date = datetime.strptime(start_date, '%d-%m-%Y')
end_date = datetime.strptime(end_date, '%d-%m-%Y')

# SQL query to update the post_generated column
update_query = '''
UPDATE hf_email
SET post_generated = 0
WHERE date(substr(email_date, 7, 4) || '-' || substr(email_date, 4, 2) || '-' || substr(email_date, 1, 2))
BETWEEN date(?) AND date(?)
'''

# Execute the update query
cursor.execute(update_query, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Update completed successfully.")