import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import plotly.graph_objects as go
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
 
 
# Load the final dataset
df = pd.read_csv("final_dataset_3_5000.csv")
 
# Connect to SQLite database (or create if it doesn't exist)
conn = sqlite3.connect("hospital_data2.db")
cursor = conn.cursor()
 
# Create the hospital_data2 table
cursor.execute("""
CREATE TABLE IF NOT EXISTS hospital_data2 ( ENCOUNTER TEXT PRIMARY KEY, START TEXT, STOP TEXT, PATIENT TEXT, ENCOUNTERCLASS TEXT, BIRTHDATE TEXT, DEATHDATE TEXT, GENDER TEXT, CITY TEXT, STATE TEXT, AGE INTEGER, HOSPITAL_NAME TEXT, ADDRESS TEXT, ORGANIZATION_ID TEXT, PROVIDER TEXT, PROVIDER_NAME TEXT, PROVIDER_GENDER TEXT, DEVICES_USED TEXT, LOS INTEGER
               );
""")
 
# Insert data (overwrite if already exists)
df.to_sql("hospital_data2", conn, if_exists="replace", index=False)
 
# ✅ Create users table for login/signup
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
);
""")
conn.commit()
conn.close()
print("✅ Database setup completed.")
 
# ✅ Authenticate user
def authenticate(username, password):
    conn = sqlite3.connect("hospital_data2.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = cursor.fetchone()
    conn.close()
    return result is not None
 
# ✅ Sign-up page
def signup():
    st.title("📝 Sign Up")
    new_user = st.text_input("Choose a Username")
    new_pass = st.text_input("Choose a Password", type="password")
 
    if st.button("Create Account"):
        conn = sqlite3.connect("hospital_data2.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (new_user,))
        if cursor.fetchone():
            st.error("Username already exists.")
        else:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, new_pass))
            conn.commit()
            st.success("Account created! Please log in.")
            st.session_state.menu = "Login"
            st.rerun()
        conn.close()
 
# ✅ Login page
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
 
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.menu = "Dashboard"
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
 
    if st.button("📝 Sign Up"):
        st.session_state.menu = "SignUp"
        st.rerun()
 
# ----------------------
# 📊 Dashboard Function
# ----------------------
# ✅ Dashboard Logic
def dashboard():
    st.title("Resource Dashboard")
 
    @st.cache_data
    def load_data():
        conn = sqlite3.connect("hospital_data2.db")
        df = pd.read_sql("SELECT * FROM hospital_data2", conn)
        conn.close()
        df["START"] = pd.to_datetime(df["START"]).dt.tz_localize(None)
        df["STOP"] = pd.to_datetime(df["STOP"]).dt.tz_localize(None)
        return df
 
    df = load_data()
 
    st.sidebar.header("Filters")
    selected_states = st.sidebar.multiselect("🌎 State", sorted(df["STATE"].dropna().unique()))
    selected_cities = st.sidebar.multiselect("🏙️ City", sorted(df["CITY"].dropna().unique()))
    hospital_options = sorted(df[df["CITY"].isin(selected_cities)]["HOSPITAL_NAME"].dropna().unique()) if selected_cities else sorted(df["HOSPITAL_NAME"].dropna().unique())
    selected_hospitals = st.sidebar.multiselect("🏥 Hospital", hospital_options)
    min_date = df["START"].min().date()
    max_date = df["START"].max().date()

    st.sidebar.markdown("#### 📅 Date Filter Mode")
    Date_range = st.sidebar.date_input( "Select a Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date )

# Unpack the start and end dates
    start_date, end_date = Date_range
 
    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df["STATE"].isin(selected_states)]
    if selected_cities:
        filtered_df = filtered_df[filtered_df["CITY"].isin(selected_cities)]
    if selected_hospitals:
        filtered_df = filtered_df[filtered_df["HOSPITAL_NAME"].isin(selected_hospitals)]
    if Date_range:
        filtered_df = filtered_df[(filtered_df["START"].dt.date >= start_date) & (filtered_df["START"].dt.date <= end_date)]
    
    st.session_state.dashboard_filters = {
        "STATE": selected_states,
        "CITY": selected_cities,
        "HOSPITAL": selected_hospitals,
        "DATE_RANGE": (start_date, end_date)
    }
 
    if filtered_df.empty:
        st.warning("⚠️ No data for selected filters.")
        return
 
    Admitted_df = filtered_df[filtered_df["LOS"] >= 1]
    tab1, tab2 = st.tabs(["🛏️ Bed Utilization", "👩‍⚕️ Staffing"])
    with tab1:
        bed_utilization_dashboard(Admitted_df)
    with tab2:
        staffing_dashboard(filtered_df)
 
# ✅ Bed Utilization Dashboard
def bed_utilization_dashboard(Admitted_df):
    st.subheader("🛏️ Bed Utilization Dashboard")
    total_encounters = Admitted_df["ENCOUNTER"].nunique()
    st.markdown(f"#### Total admissions: `{total_encounters}`")

    if Admitted_df.empty:
        st.warning("⚠️ No inpatient data for selected filters.")
        return
 
    patient_counts = Admitted_df.groupby("HOSPITAL_NAME")["ENCOUNTER"].nunique().reset_index()
    patient_counts.columns = ["HOSPITAL_NAME", "TOTAL_PATIENTS"]
    patient_counts["LABEL"] = [f"H{i+1}" for i in range(len(patient_counts))]
    patient_counts["HOVER"] = patient_counts["HOSPITAL_NAME"]
    fig1 = px.bar(patient_counts, x="LABEL", y="TOTAL_PATIENTS", hover_name="HOVER", title="Total Patients admitted per Hospital", text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)
 
    los_avg = Admitted_df.groupby("HOSPITAL_NAME")["LOS"].mean().reset_index()
    los_avg.columns = ["HOSPITAL_NAME", "AVG_LOS"]
    los_avg["LABEL"] = [f"H{i+1}" for i in range(len(los_avg))]
    los_avg["HOVER"] = los_avg["HOSPITAL_NAME"]
    fig2 = px.bar(los_avg, x="LABEL", y="AVG_LOS", hover_name="HOVER", title="Average LOS per Hospital", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)
 
    gender_hosp = Admitted_df.groupby(["HOSPITAL_NAME", "GENDER"])["ENCOUNTER"].nunique().reset_index()
    gender_hosp.columns = ["HOSPITAL_NAME", "GENDER", "PATIENT_COUNT"]
    label_map = {name: f"H{i+1}" for i, name in enumerate(gender_hosp["HOSPITAL_NAME"].unique())}
    gender_hosp["LABEL"] = gender_hosp["HOSPITAL_NAME"].map(label_map)
    fig5 = px.bar(gender_hosp, x="LABEL", y="PATIENT_COUNT", color="GENDER", barmode="stack", hover_name="HOSPITAL_NAME", title="Gender Distribution by Hospital", text_auto=True)
    st.plotly_chart(fig5, use_container_width=True)
 
    bins = [0, 18, 35, 50, 65, 80, 120]
    labels = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
    Admitted_df["AGE_GROUP"] = pd.cut(Admitted_df["AGE"], bins=bins, labels=labels, right=False)
    age_dist = Admitted_df["AGE_GROUP"].value_counts().sort_index().reset_index()
    age_dist.columns = ["AGE_GROUP", "COUNT"]
    fig6 = px.bar(age_dist, x="AGE_GROUP", y="COUNT", title="Patient Count by Age Group", text_auto=True)
    st.plotly_chart(fig6, use_container_width=True)
   
def staffing_dashboard(filtered_df):
    st.subheader("👩‍⚕️ Staffing Dashboard")
   
# 📊 1. Total Encounters by Hospital
# ----------------------------------
    encounters_per_hosp = filtered_df.groupby("HOSPITAL_NAME")["ENCOUNTER"].nunique().reset_index()
    encounters_per_hosp.columns = ["HOSPITAL_NAME", "ENCOUNTER_COUNT"]
    providers_per_hosp = filtered_df.groupby("HOSPITAL_NAME")["PROVIDER_NAME"].nunique().reset_index()
    providers_per_hosp.columns = ["HOSPITAL_NAME", "PROVIDER_COUNT"]
    fig1 = px.bar(encounters_per_hosp, x="HOSPITAL_NAME", y="ENCOUNTER_COUNT", title="📊 Total Encounters by Hospital", text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)
 
# 👩‍⚕️ Providers by Hospital
    fig2 = px.bar(providers_per_hosp, x="HOSPITAL_NAME", y="PROVIDER_COUNT", title="👩‍⚕️ Providers per Hospital", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)
 
# 📈 Avg Encounters per Provider
    merged = pd.merge(encounters_per_hosp, providers_per_hosp, on="HOSPITAL_NAME", how="inner")
    merged["AVG_ENCOUNTERS_PER_PROVIDER"] = (merged["ENCOUNTER_COUNT"] / merged["PROVIDER_COUNT"]).round(2)
    fig3 = px.bar(merged, x="HOSPITAL_NAME", y="AVG_ENCOUNTERS_PER_PROVIDER", title="📈 Avg Encounters per Provider per Hospital", text_auto=True)
    st.plotly_chart(fig3, use_container_width=True)
 
# ✅ Navigation Bar
def top_navbar():
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Dashboard"):
            st.session_state.menu = "Dashboard"
            st.rerun()
    with col2:
        if st.button("🔮 Predictions"):
            st.session_state.menu = "Predictions"
            st.rerun()
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

def bed_demand_forecast():
    st.subheader("🛏️ Forecast Bed Demand")

    # ----------------- GET FILTERS -----------------
    filters = st.session_state.get("dashboard_filters", {})
    states = filters.get("STATE", [])
    cities = filters.get("CITY", [])
    hospitals = filters.get("HOSPITAL", [])
    date_range = filters.get("DATE_RANGE", None)

    @st.cache_data
    def load_data():
        conn = sqlite3.connect("hospital_data2.db")
        df = pd.read_sql("SELECT * FROM hospital_data2", conn)
        conn.close()
        df["START"] = pd.to_datetime(df["START"]).dt.tz_localize(None)
        return df

    df = load_data()

    # ----------------- APPLY FILTERS -----------------
    if states:
        df = df[df["STATE"].isin(states)]
    if cities:
        df = df[df["CITY"].isin(cities)]
    if hospitals:
        df = df[df["HOSPITAL_NAME"].isin(hospitals)]
    if date_range:
        df = df[(df["START"].dt.date >= date_range[0]) & (df["START"].dt.date <= date_range[1])]

    df = df[df["LOS"] >= 1]
    if df.empty:
        st.warning("⚠️ No admission data available for selected filters.")
        return

    # ----------------- AGGREGATE MONTHLY -----------------
    df["GROUP_DATE"] = df["START"].dt.to_period("M").apply(lambda r: r.start_time)
    demand_df = df.groupby("GROUP_DATE")["ENCOUNTER"].count().reset_index()
    demand_df.columns = ["DATE", "BED_DEMAND"]
    demand_df = demand_df.sort_values("DATE")
    demand_df.set_index("DATE", inplace=True)

    st.write("Records after grouping the admissions monthly:", len(demand_df))

    # ----------------- TRAIN-TEST SPLIT -----------------
    split_idx = int(len(demand_df) * 0.8)
    train_series = demand_df["BED_DEMAND"].iloc[:split_idx]
    test_series = demand_df["BED_DEMAND"].iloc[split_idx:]

    # ----------------- FORECAST MODEL -----------------
    try:
        model = SARIMAX(train_series, order=(1, 0, 3), seasonal_order=(1, 1, 1, 12), 
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        eval_pred = model_fit.forecast(steps=len(test_series))
    except Exception as e:
        st.warning(f"⚠️ SARIMAX(3,1,2)(1,1,1,12) evaluation failed: {e}")
        return

    # ----------------- FINAL FORECAST -----------------
    try:
        model = SARIMAX(demand_df["BED_DEMAND"], order=(1, 0, 3), seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast_steps = 5
        future_dates = pd.date_range(
            start=demand_df.index[-1] + pd.offsets.MonthBegin(1),
            periods=forecast_steps,
            freq="MS"
        ).date

        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_df = pd.DataFrame({
            "DATE": future_dates,
            "PREDICTED_BED_DEMAND": np.round(forecast).astype(int)
        })

    except Exception as e:
        st.error(f"❌ Final SARIMAX forecast failed: {e}")
        return

    # ----------------- ACTUAL vs PREDICTED PLOT -----------------

    plt.figure(figsize=(10, 4))
    plt.plot(test_series.index, test_series, label="Actual")
    plt.plot(test_series.index, eval_pred, label="Predicted", linestyle="--")
    plt.title("📉 Actual vs Predicted Bed Demand (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Bed Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # ----------------- COMBINE HISTORICAL + FORECAST -----------------
    recent_actuals = demand_df["BED_DEMAND"].tail(7).reset_index()
    recent_actuals["Type"] = "Actual"

    forecast_df["Type"] = "Forecast"
    forecast_df.columns = ["DATE", "BED_DEMAND", "Type"]

    combined_df = pd.concat([recent_actuals, forecast_df])
    

    mae = mean_absolute_error(test_series, eval_pred)
    st.markdown(f"##### Average Forecast Error: ±{int(round(mae))} beds/month")

    # ----------------- PLOT WITH PAST + FORECAST -----------------
    fig_combined = px.line(combined_df, x="DATE", y="BED_DEMAND", color="Type",
                            title="📊 Actual + Forecasted Monthly Bed Demand",
                            markers=True)
    fig_combined.update_layout(height=500)
    st.plotly_chart(fig_combined, use_container_width=True)

def staffing_forecast():
    st.subheader("🧑‍⚕️ Forecast Monthly Staffing Needs")

    # ----------------- GET FILTERS -----------------
    filters = st.session_state.get("dashboard_filters", {})
    states = filters.get("STATE", [])
    cities = filters.get("CITY", [])
    hospitals = filters.get("HOSPITAL", [])
    date_range = filters.get("DATE_RANGE", None)

    @st.cache_data
    def load_data():
        conn = sqlite3.connect("hospital_data2.db")
        df = pd.read_sql("SELECT * FROM hospital_data2", conn)
        conn.close()
        df["START"] = pd.to_datetime(df["START"]).dt.tz_localize(None)
        return df

    df = load_data()

    # ----------------- APPLY FILTERS -----------------
    if states:
        df = df[df["STATE"].isin(states)]
    if cities:
        df = df[df["CITY"].isin(cities)]
    if hospitals:
        df = df[df["HOSPITAL_NAME"].isin(hospitals)]
    if date_range:
        df = df[(df["START"].dt.date >= date_range[0]) & (df["START"].dt.date <= date_range[1])]

    if df.empty:
        st.warning("⚠️ No data available for selected filters.")
        return

    # ----------------- AGGREGATE MONTHLY -----------------
    df["MONTH"] = df["START"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("MONTH").agg({
        "ENCOUNTER": "nunique",
        "PROVIDER_NAME": "nunique"
    }).reset_index()

    monthly.columns = ["DATE", "PATIENT_COUNT", "PROVIDER_COUNT"]
    monthly = monthly[monthly["PROVIDER_COUNT"] > 0]
    if monthly.empty:
        st.warning("⚠️ Not enough data for staffing forecast.")
        return

    monthly["RATIO"] = monthly["PATIENT_COUNT"] / monthly["PROVIDER_COUNT"]
    target_ratio = monthly["RATIO"].mean()
    st.markdown(f"📌 **Avg patient-to-provider ratio:** {int(target_ratio)}")

    monthly["REQUIRED_PROVIDERS"] = (monthly["PATIENT_COUNT"] / target_ratio)
    monthly.set_index("DATE", inplace=True)
    ts = monthly["REQUIRED_PROVIDERS"]

    # ----------------- TRAIN-TEST SPLIT -----------------
    split_idx = int(len(ts) * 0.7)
    train_series = ts.iloc[:split_idx]
    test_series = ts.iloc[split_idx:]

    # ----------------- FORECAST MODEL -----------------
    try:
        model = SARIMAX(train_series, order=(3, 1, 1), seasonal_order=(2, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        eval_pred = model_fit.forecast(steps=len(test_series))
    except Exception as e:
        st.warning(f"⚠️ SARIMAX(1,1,1)(1,1,1,12) evaluation failed: {e}")
        return

    # ----------------- FINAL FORECAST -----------------
    try:
        model = SARIMAX(ts, order=(3, 1, 1), seasonal_order=(2, 1, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast_steps = 5
        future_dates = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1),
                                     periods=forecast_steps, freq="MS").date
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_df = pd.DataFrame({
            "DATE": future_dates,
            "PREDICTED_PROVIDERS": np.round(forecast).astype(int)
        })
    except Exception as e:
        st.error(f"❌ Final SARIMAX forecast failed: {e}")
        return
    
       # ----------------- Optional: Evaluation Plot -----------------

    plt.figure(figsize=(10, 4))
    plt.plot(test_series.index, test_series, label="Actual")
    plt.plot(test_series.index, eval_pred, label="Predicted", linestyle="--")
    plt.title("📉 Actual vs Predicted Providers (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Required Providers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # ----------------- COMBINE HISTORICAL + FORECAST -----------------
    recent_actuals = ts.tail(5).reset_index()
    recent_actuals.columns = ["DATE", "REQUIRED_PROVIDERS"]
    recent_actuals["Type"] = "Actual"

    forecast_df.columns = ["DATE", "REQUIRED_PROVIDERS"]
    forecast_df["Type"] = "Forecast"

    combined_df = pd.concat([recent_actuals, forecast_df])
    
    mae = mean_absolute_error(test_series, eval_pred)
    st.markdown(f"##### Average Forecast Error: ±{int(round(mae))} providers/month")

    # ----------------- PLOT COMBINED -----------------
    fig_combined = px.line(combined_df, x="DATE", y="REQUIRED_PROVIDERS", color="Type",
                           title="📊 Actual + Forecasted Monthly Required Providers",
                           markers=True)
    fig_combined.update_layout(height=500)
    st.plotly_chart(fig_combined, use_container_width=True)


 
def predictions_page():
    st.title("Resource Forecasting")
 
    # Create Tabs
    tab1, tab2= st.tabs(["🛏️ Bed Forecast", "👩‍⚕️ Staffing Forecast"])
 
    with tab1:
        bed_demand_forecast()
 
    with tab2:
        staffing_forecast()
 
# ✅ Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "menu" not in st.session_state:
    st.session_state.menu = "Login"
 
# ✅ App Navigation Logic
if not st.session_state.logged_in:
    if st.session_state.menu == "SignUp":
        signup()
    else:
        login()
else:
    top_navbar()  # show navbar only when logged in
 
    if st.session_state.menu == "Dashboard":
        dashboard()
    elif st.session_state.menu == "Predictions":
        predictions_page()
