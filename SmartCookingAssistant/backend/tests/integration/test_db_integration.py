from __future__ import annotations

import os

import mysql.connector
import pytest


SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(255) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        last_login DATETIME NULL,
        login_count INT NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS login_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        ip_address VARCHAR(64),
        device_info VARCHAR(255),
        login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recipes (
        recipe_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        recipe_name VARCHAR(255) NOT NULL,
        meal_type VARCHAR(64),
        diet_type VARCHAR(64),
        language VARCHAR(16),
        recipe_json JSON,
        calories VARCHAR(64),
        protein VARCHAR(64),
        carbs VARCHAR(64),
        fat VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS uploaded_images (
        image_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        image_path TEXT,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detected_ingredients (
        ingredient_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        ingredients_json JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


@pytest.fixture(scope="session")
def db_conn():
    if os.getenv("RUN_DB_TESTS", "0") != "1":
        pytest.skip("DB tests disabled. Set RUN_DB_TESTS=1 to enable.")

    required = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    if any(not os.getenv(key) for key in required):
        pytest.skip("DB integration env vars are not set.")

    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )
    cursor = connection.cursor()
    for stmt in SCHEMA_SQL:
        cursor.execute(stmt)
    connection.commit()

    for table in ["login_history", "uploaded_images", "detected_ingredients", "recipes", "users"]:
        cursor.execute(f"TRUNCATE TABLE {table}")
    connection.commit()

    cursor.close()
    yield connection
    connection.close()


@pytest.mark.db
def test_signup_inserts_user(db_conn) -> None:
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
        ("DB QA", "dbqa@example.com", "hashed_pw"),
    )
    db_conn.commit()

    cursor.execute("SELECT COUNT(*) FROM users WHERE email=%s", ("dbqa@example.com",))
    count = cursor.fetchone()[0]
    cursor.close()

    assert count == 1


@pytest.mark.db
def test_login_history_insert(db_conn) -> None:
    cursor = db_conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE email=%s", ("dbqa@example.com",))
    user_id = cursor.fetchone()[0]
    cursor.execute(
        "INSERT INTO login_history (user_id, ip_address, device_info) VALUES (%s, %s, %s)",
        (user_id, "127.0.0.1", "pytest"),
    )
    db_conn.commit()

    cursor.execute("SELECT COUNT(*) FROM login_history WHERE user_id=%s", (user_id,))
    count = cursor.fetchone()[0]
    cursor.close()

    assert count >= 1


@pytest.mark.db
def test_recipe_and_image_persist(db_conn) -> None:
    cursor = db_conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE email=%s", ("dbqa@example.com",))
    user_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO recipes
        (user_id, recipe_name, meal_type, diet_type, language, recipe_json, calories, protein, carbs, fat)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (user_id, "Tomato Bath", "lunch", "veg", "en", "{}", "300", "8", "45", "10"),
    )
    cursor.execute(
        "INSERT INTO uploaded_images (user_id, image_path) VALUES (%s, %s)",
        (user_id, "/tmp/test.jpg"),
    )
    db_conn.commit()

    cursor.execute("SELECT COUNT(*) FROM recipes WHERE user_id=%s", (user_id,))
    recipes_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM uploaded_images WHERE user_id=%s", (user_id,))
    images_count = cursor.fetchone()[0]
    cursor.close()

    assert recipes_count >= 1
    assert images_count >= 1
