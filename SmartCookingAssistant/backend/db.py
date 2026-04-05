import os

import mysql.connector


def get_db_connection():
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        raise RuntimeError("DB_PASSWORD is not set. Add it to your .env file.")

    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=db_password,
        database=os.getenv("DB_NAME", "smart_cooking_assistant"),
    )
