#!/usr/bin/python

import psycopg2

conn = psycopg2.connect(database="ncstate", user = "ncstate", password = "ohToapiegei7", host = "13.84.183.46", port = "5432")

cur = conn.cursor()
cur.execute("SELECT Current  from ncstate")

rows = cur.fetchall()


for row in rows:
	print row
conn.close()

print "Opened database successfully"