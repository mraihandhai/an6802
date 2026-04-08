from flask import Flask, render_template, request, session
import joblib, os, json
from groq import Groq
import numpy as np

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq()

# ── Food expenditure model (existing) ────────────────────────────────────────
food_model = joblib.load("foodexp.pkl")

app = Flask(__name__)
app.secret_key = "an6802-secret-key-2025"

# =============================================================================
#  TOOL DEFINITIONS  (used by the Agentic AI workflow)
# =============================================================================

def tool_fetch_stock_data(ticker: str) -> dict:
    """Tool 1: Fetch stock price history via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")
        if hist.empty:
            raise ValueError("No data returned")
        closes = hist["Close"].tolist()
        dates  = [str(d.date()) for d in hist.index]
        return {
            "success": True,
            "ticker": ticker,
            "dates": dates[-30:],
            "closes": closes[-30:],
            "current_price": round(closes[-1], 2),
            "price_6mo_ago": round(closes[0], 2),
            "data_points": len(closes),
            "_all_closes": closes,
        }
    except Exception:
        import random, math
        random.seed(hash(ticker) % 9999)
        base = 35.0 if "D05" in ticker else 30.0
        closes = []
        for i in range(130):
            noise = random.gauss(0, 0.4)
            trend = 0.005 * math.sin(i / 20)
            base  = max(10, base * (1 + trend + noise / 100))
            closes.append(round(base, 2))
        from datetime import date, timedelta
        start = date.today() - timedelta(days=129)
        dates = [(start + timedelta(days=i)).isoformat() for i in range(130)]
        return {
            "success": True,
            "ticker": ticker,
            "dates": dates[-30:],
            "closes": closes[-30:],
            "current_price": round(closes[-1], 2),
            "price_6mo_ago": round(closes[0], 2),
            "data_points": len(closes),
            "note": "Simulated data (yfinance blocked in sandbox; live data works on deployment)",
            "_all_closes": closes,
        }


def tool_compute_risk_metrics(closes: list) -> dict:
    """Tool 2: Compute financial risk metrics from price series."""
    arr     = np.array(closes, dtype=float)
    returns = np.diff(arr) / arr[:-1]

    daily_vol     = float(np.std(returns))
    annual_vol    = daily_vol * np.sqrt(252)
    mean_return   = float(np.mean(returns))
    annual_return = mean_return * 252
    risk_free     = 0.04 / 252
    sharpe        = (mean_return - risk_free) / daily_vol * np.sqrt(252) if daily_vol > 0 else 0
    var_95        = float(np.percentile(returns, 5))
    max_dd        = _max_drawdown(arr)

    if annual_vol < 0.15:
        risk_band = "Low"
    elif annual_vol < 0.30:
        risk_band = "Medium"
    else:
        risk_band = "High"

    return {
        "daily_volatility":  round(daily_vol * 100, 4),
        "annual_volatility": round(annual_vol * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "var_95_pct":        round(var_95 * 100, 4),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "risk_band":         risk_band,
    }


def _max_drawdown(prices: np.ndarray) -> float:
    peak   = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def tool_arima_forecast(closes: list, steps: int = 5) -> dict:
    """Tool 3: Fit ARIMA(2,1,2) and forecast next N days."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings("ignore")
        series = closes[-60:]
        model  = ARIMA(series, order=(2, 1, 2))
        fit    = model.fit()
        fc     = fit.forecast(steps=steps)
        conf   = fit.get_forecast(steps=steps).conf_int(alpha=0.10)
        return {
            "success":  True,
            "model":    "ARIMA(2,1,2)",
            "forecast": [round(float(v), 2) for v in fc],
            "lower_90": [round(float(v), 2) for v in conf.iloc[:, 0]],
            "upper_90": [round(float(v), 2) for v in conf.iloc[:, 1]],
            "steps":    steps,
        }
    except Exception:
        arr   = np.array(closes[-10:])
        slope = float(np.polyfit(range(len(arr)), arr, 1)[0])
        base  = float(arr[-1])
        fc    = [round(base + slope * (i + 1), 2) for i in range(steps)]
        return {
            "success":  True,
            "model":    "Linear Trend (ARIMA fallback)",
            "forecast": fc,
            "lower_90": [round(v * 0.97, 2) for v in fc],
            "upper_90": [round(v * 1.03, 2) for v in fc],
            "steps":    steps,
        }


def tool_news_sentiment(ticker: str, company: str) -> dict:
    """Tool 4: Use Groq LLM to assess news sentiment for the stock."""
    prompt = (
        f"You are a financial news analyst. Based on general knowledge about {company} ({ticker}), "
        f"provide a brief sentiment assessment. Reply ONLY with a JSON object with keys: "
        f"sentiment (Positive/Neutral/Negative), score (float -1.0 to 1.0), "
        f"key_factors (list of 3 short strings), outlook (one sentence). No markdown, no code block."
    )
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        raw  = r.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        data["success"] = True
        return data
    except Exception:
        return {
            "success":     True,
            "sentiment":   "Neutral",
            "score":       0.0,
            "key_factors": ["Market conditions", "Sector performance", "Macro environment"],
            "outlook":     "Insufficient data for confident sentiment assessment.",
        }


# =============================================================================
#  AGENTIC AI WORKFLOW
# =============================================================================

TICKER_MAP = {
    "DBS": {"ticker": "D05.SI", "company": "DBS Group Holdings"},
    "UOB": {"ticker": "U11.SI", "company": "United Overseas Bank"},
}


def run_agent(stock_choice: str) -> dict:
    """
    Multi-step agentic workflow:
      Step 1 - Plan
      Step 2 - Fetch stock data         (Tool 1: yfinance)
      Step 3 - Compute risk metrics     (Tool 2: numpy/stats)
      Step 4 - ARIMA forecast           (Tool 3: statsmodels)
      Step 5 - News sentiment           (Tool 4: Groq LLM)
      Step 6 - Synthesise recommendation (LLM)
    """
    info    = TICKER_MAP[stock_choice]
    ticker  = info["ticker"]
    company = info["company"]
    steps_log = []

    # Step 1
    steps_log.append({
        "step": 1, "name": "Agent Planning",
        "description": (
            f"Agent identified task: analyse {company} ({ticker}). "
            f"Planned tool sequence: fetch_stock_data -> compute_risk_metrics "
            f"-> arima_forecast -> news_sentiment -> synthesise."
        ),
    })

    # Step 2
    stock_data = tool_fetch_stock_data(ticker)
    closes_all = stock_data.get("_all_closes", stock_data["closes"])
    steps_log.append({
        "step": 2, "name": "Tool 1: Fetch Stock Data",
        "description": (
            f"Retrieved {stock_data['data_points']} trading days of price data. "
            f"Current price: SGD {stock_data['current_price']}. "
            + (f"Note: {stock_data['note']}" if "note" in stock_data else "Live data from Yahoo Finance.")
        ),
    })

    # Step 3
    risk = tool_compute_risk_metrics(closes_all)
    steps_log.append({
        "step": 3, "name": "Tool 2: Risk Metrics Computation",
        "description": (
            f"Annual volatility: {risk['annual_volatility']}% | "
            f"Sharpe Ratio: {risk['sharpe_ratio']} | "
            f"95% 1-day VaR: {risk['var_95_pct']}% | "
            f"Max Drawdown: {risk['max_drawdown_pct']}% | "
            f"Risk Band: {risk['risk_band']}."
        ),
    })

    # Step 4
    forecast = tool_arima_forecast(closes_all, steps=5)
    steps_log.append({
        "step": 4, "name": "Tool 3: ARIMA Price Forecast",
        "description": (
            f"Model fitted: {forecast['model']}. "
            f"5-day price forecast (SGD): {forecast['forecast']}. "
            f"90% confidence interval upper bound: {forecast['upper_90']}."
        ),
    })

    # Step 5
    sentiment = tool_news_sentiment(ticker, company)
    steps_log.append({
        "step": 5, "name": "Tool 4: News Sentiment Analysis",
        "description": (
            f"Sentiment: {sentiment['sentiment']} (score: {sentiment['score']}). "
            f"Key factors: {', '.join(sentiment.get('key_factors', []))}. "
            f"Outlook: {sentiment.get('outlook', 'N/A')}"
        ),
    })

    # Step 6
    synthesis_prompt = (
        f"You are a professional financial analyst complying with MAS (Monetary Authority of Singapore) guidelines. "
        f"Synthesise the following quantitative data for {company} and provide a concise investment recommendation "
        f"in 3-4 sentences. Mention model limitations and regulatory context briefly. "
        f"End with a clear Buy / Hold / Sell signal. "
        f"Data: current price SGD {stock_data['current_price']}, "
        f"annual volatility {risk['annual_volatility']}%, "
        f"Sharpe ratio {risk['sharpe_ratio']}, "
        f"max drawdown {risk['max_drawdown_pct']}%, "
        f"risk band {risk['risk_band']}, "
        f"5-day ARIMA forecast {forecast['forecast']}, "
        f"news sentiment {sentiment['sentiment']} (score {sentiment['score']})."
    )
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=400,
        )
        recommendation = r.choices[0].message.content.strip()
    except Exception:
        recommendation = (
            f"Based on the quantitative analysis, {company} shows {risk['risk_band'].lower()} risk "
            f"with a Sharpe ratio of {risk['sharpe_ratio']}. "
            f"The ARIMA forecast projects SGD {forecast['forecast'][-1]} in 5 days. "
            f"News sentiment is {sentiment['sentiment']}. "
            f"Note: This is model-generated and should not replace professional financial advice. Signal: Hold."
        )

    steps_log.append({
        "step": 6, "name": "Agent Synthesis & Recommendation",
        "description": recommendation,
    })

    return {
        "company":        company,
        "ticker":         ticker,
        "stock_data":     stock_data,
        "risk":           risk,
        "forecast":       forecast,
        "sentiment":      sentiment,
        "steps_log":      steps_log,
        "recommendation": recommendation,
    }


# =============================================================================
#  FLASK ROUTES
# =============================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    q = request.form.get("q", "")
    session["username"] = q
    return render_template("main.html", username=q)

@app.route("/ethics", methods=["GET", "POST"])
def ethics():
    return render_template("ethics.html")

@app.route("/correct", methods=["GET", "POST"])
def correct():
    return render_template("correct.html")

@app.route("/wrong", methods=["GET", "POST"])
def wrong():
    return render_template("wrong.html")

@app.route("/econ", methods=["GET", "POST"])
def econ():
    return render_template("econ.html")

@app.route("/foodExp", methods=["GET", "POST"])
def foodExp():
    q = float(request.form.get("q"))
    r = food_model.predict([[q]])
    return render_template("foodExp.html", r=round(float(r[0]), 2))

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    return render_template("chatbot.html")

@app.route("/roe", methods=["GET", "POST"])
def roe():
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": "Please explain what is ROE in 20 words"}],
    )
    return render_template("roe.html", r=r.choices[0].message.content)

@app.route("/generalQuestion", methods=["GET", "POST"])
def generalQuestion():
    return render_template("generalQuestion.html")

@app.route("/groqReply", methods=["GET", "POST"])
def groqReply():
    q = request.form.get("q")
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": q}],
    )
    return render_template("groqReply.html", r=r.choices[0].message.content)

@app.route("/equity", methods=["GET", "POST"])
def equity():
    return render_template("equity.html")

@app.route("/apple", methods=["GET", "POST"])
def apple():
    return render_template("apple.html")

# ── NEW ROUTES ────────────────────────────────────────────────────────────────

@app.route("/stock", methods=["GET", "POST"])
def stock():
    return render_template("stock.html")

@app.route("/agent", methods=["GET", "POST"])
def agent():
    stock_choice = request.form.get("stock", "DBS")
    if stock_choice not in TICKER_MAP:
        stock_choice = "DBS"
    result = run_agent(stock_choice)
    return render_template("agent.html", result=result)

@app.route("/risk", methods=["GET", "POST"])
def risk():
    return render_template("risk.html")

@app.route("/riskResult", methods=["GET", "POST"])
def riskResult():
    raw = request.form.get("prices", "")
    try:
        closes = [float(x.strip()) for x in raw.replace(",", "\n").split("\n") if x.strip()]
        if len(closes) < 10:
            raise ValueError("Please enter at least 10 price data points.")
        metrics = tool_compute_risk_metrics(closes)
        error   = None
    except Exception as e:
        metrics = None
        error   = str(e)
    return render_template("riskResult.html", metrics=metrics, error=error)

@app.route("/governance", methods=["GET", "POST"])
def governance():
    return render_template("governance.html")

if __name__ == "__main__":
    app.run(debug=True)
