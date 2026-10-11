
from pathlib import Path
import os
import sqlite3
import secrets
import hashlib
import hmac

import joblib
import pandas as pd

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware


# --------------------------------------------------
# 1. PROJECT SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATABASE_PATH = BASE_DIR / "users.db"

app = FastAPI(title="TNEB Electricity Prediction Website")

# For local development, this fallback is convenient.
# Set SESSION_SECRET to a long random secret before deployment.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get(
        "SESSION_SECRET", "replace-this-with-a-random-secret"
    ),
    same_site="lax",
    https_only=False
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --------------------------------------------------
# 2. LOAD THE EXISTING MACHINE LEARNING MODELS
# --------------------------------------------------

usage_model = joblib.load(BASE_DIR / "usage_model.pkl")
bill_model = joblib.load(BASE_DIR / "bill_model.pkl")


# --------------------------------------------------
# 3. DATABASE SETUP
# --------------------------------------------------

def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_users_table():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL
            )
        """)


create_users_table()


# --------------------------------------------------
# 4. PASSWORD SECURITY
# --------------------------------------------------

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1
    )

    return salt, password_hash


def verify_password(password, salt, saved_hash):
    _, entered_hash = hash_password(password, salt)
    return hmac.compare_digest(entered_hash, saved_hash)


def logged_in(request: Request):
    return request.session.get("user_id") is not None


# --------------------------------------------------
# 5. HOME PAGE
# --------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not logged_in(request):
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )



# --------------------------------------------------
# 6. ABOUT PAGE
# --------------------------------------------------

@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={}
    )


# --------------------------------------------------
# 7. SIGNUP PAGE
# --------------------------------------------------

@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    if logged_in(request):
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={"error": None}
    )


@app.post("/signup", response_class=HTMLResponse)
def signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
    
):
    username = username.strip()

    error = None

    if len(username) < 3 or len(username) > 30:
        error = "Username must contain 3 to 30 characters."

    elif any(char.isspace() for char in username):
        error = "Username cannot contain spaces."

    elif len(password) < 8:
        error = "Password must contain at least 8 characters."

    elif len(password) > 128:
        error = "Password is too long."

    elif password != confirm_password:
        error = "Passwords do not match."

    if error:
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": error},
            status_code=400
        )

    salt, password_hash = hash_password(password)

    try:
        with get_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users (username, salt, password_hash)
                VALUES (?, ?, ?)
                """,
                (username, salt, password_hash)
            )
            user_id = cursor.lastrowid

    except sqlite3.IntegrityError:
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={"error": "This username already exists."},
            status_code=400
        )

    request.session.clear()
    request.session["user_id"] = user_id
    request.session["username"] = username

    return RedirectResponse("/", status_code=303)


# --------------------------------------------------
# 8. LOGIN PAGE
# --------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if logged_in(request):
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None}
    )


@app.post("/login", response_class=HTMLResponse)
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    username = username.strip()

    with get_db() as db:
        user = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

    if user is None or not verify_password(
        password,
        user["salt"],
        user["password_hash"]
    ):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Incorrect username or password."},
            status_code=401
        )

    request.session.clear()
    request.session["user_id"] = user["id"]
    request.session["username"] = user["username"]

    return RedirectResponse("/", status_code=303)


# --------------------------------------------------
# 9. LOGOUT
# --------------------------------------------------

@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# --------------------------------------------------
# 10. ELECTRICITY PREDICTION PAGE
# --------------------------------------------------

@app.get("/prediction", response_class=HTMLResponse)
def prediction_page(request: Request):
    if not logged_in(request):
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={}
    )


# --------------------------------------------------
# 11. PREDICT ELECTRICITY USAGE AND BILL
# --------------------------------------------------

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    temperature: float = Form(...),
    humidity: float = Form(...),
    rainfall: float = Form(...),
    fan_usage: float = Form(...),
    washing_machine: float = Form(...),
    refrigerator_hours: float = Form(...),
    tv_usage: float = Form(...),
    ac_usage: float = Form(...),
    previous_usage: float = Form(...)
):
    if not logged_in(request):
        return RedirectResponse("/login", status_code=303)

    # Validate the submitted values.
    values = [
        temperature,
        humidity,
        rainfall,
        fan_usage,
        washing_machine,
        previous_usage
    ]

    import math

    if not all(math.isfinite(value) for value in values):
        return HTMLResponse(
            "Please enter valid numeric values.",
            status_code=400
        )

    if not 0 <= temperature <= 60:
        return HTMLResponse(
            "Temperature must be between 0 and 60 degrees C.",
            status_code=400
        )

    if not 0 <= humidity <= 100:
        return HTMLResponse(
            "Humidity must be between 0 and 100 percent.",
            status_code=400
        )

    if rainfall < 0 or previous_usage < 0:
        return HTMLResponse(
            "Rainfall and previous usage cannot be negative.",
            status_code=400
        )

    if not 0 <= fan_usage <= 24:
        return HTMLResponse(
            "Fan usage must be between 0 and 24 hours.",
            status_code=400
        )

    if not 0 <= washing_machine <= 24:
        return HTMLResponse(
            "Washing machine usage must be between 0 and 24 hours.",
            status_code=400
        )

    # IMPORTANT:
    # These column names and their order must match
    # the features used when training your saved models.
    input_data = pd.DataFrame([{
    "Temperature_C": temperature,
    "Humidity_Percent": humidity,
    "Rainfall_mm": rainfall,
    "Fan_Usage_Hours": fan_usage,
    "Washing_Machine_Usage_Hours": washing_machine,
    "Refrigerator_Hours": refrigerator_hours,
    "TV_Usage_Hours": tv_usage,
    "AC_Usage_Hours": ac_usage,
    "Previous_Month_Usage_kWh": previous_usage
}])

    predicted_usage = float(
        usage_model.predict(input_data)[0]
    )

    predicted_bill = float(
        bill_model.predict(input_data)[0]
    )

    # Prevent negative displayed predictions.
    predicted_usage = max(0.0, predicted_usage)
    predicted_bill = max(0.0, predicted_bill)
    # Save the latest prediction for the dashboard
    request.session["latest_prediction"] = {
    "usage": round(float(predicted_usage), 2),
    "bill": round(float(predicted_bill), 2),
    "previous_usage": round(float(previous_usage), 2)
}


    difference = predicted_usage - previous_usage

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "usage": round(predicted_usage, 2),
            "bill": round(predicted_bill, 2),
            "previous_usage": round(previous_usage, 2),
            "difference": round(difference, 2)
        }
    )

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    if not logged_in(request):
        return RedirectResponse("/login", status_code=303)

    # Display the latest prediction saved in this user's session
    latest = request.session.get("latest_prediction", {})

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "latest": latest,
            "username": request.session.get("username", "User")
        }
    )


# --------------------------------------------------
# MACHINE LEARNING PAGE
# --------------------------------------------------

@app.get("/machine-learning", response_class=HTMLResponse)
def machine_learning_page(request: Request):
    if not logged_in(request):
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="machine_learning.html",
        context={
            "username": request.session.get("username", "User"),
            "usage_model_loaded": usage_model is not None,
            "bill_model_loaded": bill_model is not None
        }
    )


def calculate_appliance_usage(
    refrigerator_hours,
    tv_hours,
    ac_hours,
    freezer_hours
):
    # Estimated power consumption in watts
    refrigerator_watts = 150
    tv_watts = 100
    ac_watts = 1500
    freezer_watts = 200

    # Calculate daily electricity consumption in kWh
    refrigerator_units = (
        refrigerator_watts * refrigerator_hours
    ) / 1000

    tv_units = (tv_watts * tv_hours) / 1000

    ac_units = (ac_watts * ac_hours) / 1000

    freezer_units = (
        freezer_watts * freezer_hours
    ) / 1000

    total_daily_units = (
        refrigerator_units
        + tv_units
        + ac_units
        + freezer_units
    )

    total_monthly_units = total_daily_units * 30

    return {
        "refrigerator_units": round(refrigerator_units, 2),
        "tv_units": round(tv_units, 2),
        "ac_units": round(ac_units, 2),
        "freezer_units": round(freezer_units, 2),
        "total_daily_units": round(total_daily_units, 2),
        "total_monthly_units": round(total_monthly_units, 2)
    }


# Example prediction
result = calculate_appliance_usage(
    refrigerator_hours=24,
    tv_hours=5,
    ac_hours=6,
    freezer_hours=24
)

for appliance, units in result.items():
    print(f"{appliance}: {units} kWh")