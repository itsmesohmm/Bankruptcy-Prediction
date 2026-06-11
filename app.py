from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("bankruptcy_xgb.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Raw Inputs

    data = {
        'year': float(request.form['year']),
        'Current assets': float(request.form['current_assets']),
        'Cost of goods sold': float(request.form['cogs']),
        'Depreciation and amortization': float(request.form['depreciation']),
        'EBITDA': float(request.form['ebitda']),
        'Inventory': float(request.form['inventory']),
        'Net Income': float(request.form['net_income']),
        'Total Receivables': float(request.form['receivables']),
        'Market value': float(request.form['market_value']),
        'Net sales': float(request.form['net_sales']),
        'Total assets': float(request.form['total_assets']),
        'Total Long-term debt': float(request.form['long_term_debt']),
        'EBIT': float(request.form['ebit']),
        'Gross Profit': float(request.form['gross_profit']),
        'Total Current Liabilities': float(request.form['current_liabilities']),
        'Retained Earnings': float(request.form['retained_earnings']),
        'Total Revenue': float(request.form['revenue']),
        'Total Liabilities': float(request.form['liabilities']),
        'Total Operating Expenses': float(request.form['operating_expenses'])
    }

    df = pd.DataFrame([data])

    # Feature Engineering

    df['debt_ratio'] = (
        df['Total Liabilities']
        / (df['Total assets'] + 1)
    )

    df['profit_margin'] = (
        df['Net Income']
        / (df['Total Revenue'] + 1)
    )

    df['expense_ratio'] = (
        df['Total Operating Expenses']
        / (df['Total Revenue'] + 1)
    )

    df['asset_turnover'] = (
        df['Net sales']
        / (df['Total assets'] + 1)
    )

    df['debt_earnings_ratio'] = (
        df['Total Liabilities']
        / (df['Retained Earnings'] + 1)
    )

    # Prediction

    probability = model.predict_proba(df)[0][1]

    probability_percent = round(
        probability * 100,
        2
    )

    # Classification

    if probability >= 0.5:
        prediction = "Likely Bankrupt"
    else:
        prediction = "Likely Non-Bankrupt"

    # Risk Bands

    if probability_percent < 30:

        risk = "LOW"
        color = "success"

        message = (
            "The company appears financially stable "
            "based on the provided information."
        )

    elif probability_percent < 60:

        risk = "MODERATE"
        color = "warning"

        message = (
            "The company shows some indicators "
            "associated with bankruptcy risk."
        )

    else:

        risk = "HIGH"
        color = "danger"

        message = (
            "The company shows strong indicators "
            "associated with bankruptcy."
        )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability_percent,
        risk=risk,
        color=color,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)