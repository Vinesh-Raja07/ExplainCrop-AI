import sqlite3
conn = sqlite3.connect('data/optic_crop.db')
cursor = conn.cursor()
cursor.execute("UPDATE users SET role = 'Admin'")
conn.commit()
conn.close()
print("Updated all users to Admin")
