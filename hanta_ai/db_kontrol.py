import sqlite3

conn = sqlite3.connect("hanta_ai.db")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("TABLOLAR:")
print(cur.fetchall())

print("\nRISK KAYITLARI:")
cur.execute("SELECT * FROM risk_kaydi")
kayitlar = cur.fetchall()

for kayit in kayitlar:
    print(kayit)

print("\nToplam kayıt:", len(kayitlar))

conn.close()