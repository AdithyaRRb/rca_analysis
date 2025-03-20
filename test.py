import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
from prophet import Prophet

classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

st.title("AI-Driven RCA Predictive Analysis")

uploaded_file = st.file_uploader("Upload RCA Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Compute Metrics
    df['Ticket_Resolution_Rate'] = (df['Closed_Tickets'] / df['Total_Tickets']) * 100
    df['Average_Resolution_Time'] = df['Total_Time_Taken'] / df['Closed_Tickets']
    df['Response_Time_Efficiency'] = df['Time_Acknowledged'] / df['SLA_Response_Time']
    df['Pending_Ticket_Impact'] = (df['Ongoing_Tickets'] / df['Total_Tickets']) * 100
    df['Agent_Productivity'] = df['Tickets_Handled'] / df['Agent_Shift']
    df['Backlog_Rate'] = (df['Open_Tickets'] / df['Total_Tickets']) * 100

    st.subheader("Processed RCA Metrics")
    st.dataframe(df)

    # Ticket Resolution Rate Trends
    fig1 = px.line(df, x="Date", y="Ticket_Resolution_Rate", title="Ticket Resolution Rate Over Time")
    st.plotly_chart(fig1)

    # Severity-Based Risk Analysis
    fig2 = px.scatter(df, x="Average_Resolution_Time", y="Pending_Ticket_Impact",
                      color="Severity", size="Backlog_Rate",
                      title="Incident Severity vs. Pending Ticket Impact")
    st.plotly_chart(fig2)

    # **Future Predictions Using Prophet**
    st.subheader("Predicting Future Ticket Resolution Rate")
    
    # Prepare data for Prophet
    prophet_df = df[["Date", "Ticket_Resolution_Rate"]].rename(columns={"Date": "ds", "Ticket_Resolution_Rate": "y"})
    
    model = Prophet()
    model.fit(prophet_df)
    
    # Create future dates
    future = model.make_future_dataframe(periods=30)  
    forecast = model.predict(future)
    
    fig3 = px.line(forecast, x="ds", y="yhat", title="Predicted Ticket Resolution Rate for Next 30 Days")
    st.plotly_chart(fig3)

st.subheader("Incident Severity Prediction")
incident = st.text_area("Enter an incident description for AI analysis")

if st.button("Analyze Incident"):
    if incident:
        prediction = classifier(incident)
        st.write("Predicted Category:", prediction)
    else:
        st.error("Please enter an incident description.")
