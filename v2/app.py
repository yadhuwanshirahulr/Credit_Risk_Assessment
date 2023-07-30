import joblib
import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("Credit Application Form")
    
    Occupation = st.selectbox("Occupation", options=["Accountant", "Architect", "Developer", "Doctor", "Engineer",
                                                    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
                                                    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"])
    Payment_Behaviour = st.selectbox("Payment Behavior",
                                     options=["High_spent_Large_value_payments", "High_spent_Medium_value_payments",
                                              "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
                                              "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"])


    Month = st.select_slider("Month", options=[1, 2, 3, 4, 5, 6, 7, 8])
    Num_Bank_Accounts = st.select_slider("Number of Bank Accounts", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Num_Credit_Card = st.select_slider("Number of Credit Cards", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    Num_of_Loan = st.select_slider("Number of Loan", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Interest_Rate = st.select_slider("Interest Rate", np.arange(1, 35, 1))
    # Interest_Rate = st.number_input("Interest Rate", min_value=1, max_value=34)

    Age = st.number_input("Age")
    Annual_Income = st.number_input("Annual Income", value=1.0)
    Monthly_Inhand_Salary = st.number_input("Monthly Inhand Salary")

    Num_of_Delayed_Payment = st.number_input("Number of Delayed Payments", format='%u', value=0)
    Changed_Credit_Limit = st.number_input("Changed Credit Limit")
    Num_Credit_Inquiries = st.number_input("Number of Credit Inquiries", format='%u', value=0)
    Credit_History_Age = st.number_input("Credit History Age")

    Outstanding_Debt = st.number_input("Outstanding Debt")
    Credit_Utilization_Ratio = st.number_input("Credit Utilization Ratio")
    Total_EMI_per_month = st.number_input("Total EMI per Month")
    Amount_invested_monthly = st.number_input("Amount Invested Monthly")
    Monthly_Balance = st.number_input("Monthly_Balance")

    Credit_Mix = st.radio("Credit Mix", options=("Bad", "Good", "Standard"), index=1)

    submit_button = st.button("Submit")

    if submit_button:
        # Process the form data
        process_form_data(Month, Age, Occupation, Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts,
                          Num_Credit_Card, Interest_Rate, Num_of_Loan,
                          Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix,
                          Payment_Behaviour, Credit_History_Age, Outstanding_Debt,
                          Credit_Utilization_Ratio, Total_EMI_per_month, Amount_invested_monthly, Monthly_Balance)

def process_form_data(Month, Age, Occupation, Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts,
                      Num_Credit_Card, Interest_Rate, Num_of_Loan, Num_of_Delayed_Payment,
                      Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix, Payment_Behaviour,
                      Credit_History_Age, Outstanding_Debt, Credit_Utilization_Ratio, Total_EMI_per_month,
                      Amount_invested_monthly, Monthly_Balance):

    # Feature Engineering

    Monthly_Savings = Monthly_Inhand_Salary - Annual_Income / 12
    Total_Accounts = int(Num_Bank_Accounts) + int(Num_Credit_Card)
    Savings_to_Income_Ratio = Monthly_Savings / Annual_Income

    encoders = joblib.load('v2\encoders.joblib')
    model = joblib.load('v2\Random_Forest.joblib')


    dataDict = {"Month": Month, 'Occupation': Occupation, 'Num_Bank_Accounts': Num_Bank_Accounts, 'Num_Credit_Card': Num_Credit_Card,
          'Interest_Rate': Interest_Rate, 'Num_of_Loan': Num_of_Loan, 'Credit_Mix': Credit_Mix, 'Credit_History_Age': Credit_History_Age,
          'Num_of_Delayed_Payment': Num_of_Delayed_Payment, 'Payment_of_Min_Amount': Num_of_Delayed_Payment,
          'Payment_Behaviour': Payment_Behaviour, 'Age': Age, 'Annual_Income': Annual_Income, 'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
          'Outstanding_Debt': Outstanding_Debt, 'Credit_Utilization_Ratio': Credit_Utilization_Ratio,'Changed_Credit_Limit': Changed_Credit_Limit,
          'Num_Credit_Inquiries': Num_Credit_Inquiries, 'Total_EMI_per_month': Total_EMI_per_month, 'Amount_invested_monthly': Amount_invested_monthly,
          'Monthly_Balance': Monthly_Balance, 'Total_Accounts': Total_Accounts, 'Savings_to_Income_Ratio': Savings_to_Income_Ratio}

    df = pd.DataFrame([dataDict])
    cols = ['Month', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan']
    df[cols] = df[cols].astype(int)

    decodedData = df.copy()
    for col in ['Occupation', 'Payment_Behaviour', 'Credit_Mix']:
        encoder = encoders[col]
        decodedData[col] = encoder.transform(decodedData[col])

    result = model.predict(decodedData)
    result = encoders['Credit_Score'].inverse_transform(result)

    st.write(df)
    st.write(decodedData)
    st.write(result)


if __name__ == "__main__":
    main()
