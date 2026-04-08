# AN6802 Financial WebApp

A comprehensive fintech education platform combining machine learning, time-series forecasting, LLM integration, and financial analytics — built with Flask and designed for interactive financial education.

---

## 🌐 Live Demo

> **[https://an6802-p495.onrender.com](https://an6802-p495.onrender.com)**

The application is live and publicly accessible. No installation required to try it out.

> **Note:** Hosted on Render's free tier. If the instance has been idle, the first load may take 30–60 seconds to spin up.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Agentic AI Workflow](#agentic-ai-workflow)
- [Deployment](#deployment)

---

## Overview

AN6802 Financial WebApp is an educational platform built for students and learners exploring financial analysis, AI-driven decision-making, and quantitative risk assessment. It integrates real market data, pre-trained ML models, and large language models into a cohesive, browser-based experience.

---

## Features

### 1. User Entry
A welcome screen collects the user's name before routing them to the main dashboard.

### 2. Dashboard Hub
A central navigation page providing access to all modules — educational content, predictive tools, stock analysis, and the AI chatbot.

### 3. Educational Content
- **Ethical Reasoning Test** — True/false quiz on business ethics scenarios with immediate correct/incorrect feedback.
- **Economics Concepts** — Explanatory content covering core economic principles.
- **AI Governance & Ethics** — Coverage of responsible AI use and regulatory considerations in finance.

### 4. Predictive Analytics

#### Food Expenditure Prediction
- Input: user salary
- Output: estimated weekly food expenditure
- Powered by a pre-trained scikit-learn regression model (`foodexp.pkl`) loaded via `joblib`

#### Risk Calculator
- Input: user-supplied price data
- Output: key risk metrics including:
  - Value at Risk (VaR, 95%)
  - Sharpe Ratio
  - Maximum Drawdown
  - Annual Volatility

### 5. Stock Analysis System

#### Google & Apple Enquiry Pages
Static enquiry pages with links to external Gradio-hosted stock analysis tools, opening in new browser tabs.

#### Agentic AI Workflow — DBS & UOB
A fully autonomous multi-step analysis pipeline triggered by stock selection:

| Step | Action |
|---|---|
| 1 | Fetch 6-month historical price data via `yfinance` |
| 2 | Compute risk metrics (volatility, Sharpe ratio, VaR, max drawdown, risk band) |
| 3 | Run ARIMA(2,1,2) 5-day price forecast with 90% confidence interval |
| 4 | LLM-based news sentiment analysis (Positive / Neutral / Negative) |
| 5 | Generate MAS-compliant Buy / Hold / Sell recommendation via Groq LLM |

Results are presented in a structured dashboard with metric cards, forecast pills, sentiment badges, and a full agent reasoning log.

### 6. AI Financial Chatbot
- Powered by the Groq API (`llama-3.1-8b-instant`)
- Supports predefined financial concept explanations (e.g., ROE, Sharpe Ratio)
- Supports open-ended user questions with plain-text, concise AI responses
- System-prompted to avoid markdown formatting for clean in-browser display

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Production Server | Gunicorn |
| Frontend | HTML, Jinja2, CSS |
| Fonts | Google Fonts (DM Sans, Libre Baskerville) |
| ML Model | scikit-learn, joblib |
| Time-Series Forecasting | statsmodels (ARIMA) |
| Market Data | yfinance |
| Data Processing | NumPy, Pandas |
| LLM / Chatbot | Groq API — `llama-3.1-8b-instant` |

---

## Project Structure

```
an6802/
├── app.py                    # Flask app — all routes and business logic
├── requirements.txt          # Python dependencies
├── foodexp.pkl               # Pre-trained food expenditure ML model
├── static/
│   └── styles.css            # Global stylesheet (Claude-inspired design system)
└── templates/                # 18 Jinja2 HTML templates
    ├── index.html            # Welcome / name entry
    ├── main.html             # Dashboard navigation hub
    ├── ethics.html           # Ethics quiz
    ├── correct.html          # Ethics — correct answer feedback
    ├── wrong.html            # Ethics — incorrect answer feedback
    ├── econ.html             # Economics concepts
    ├── equity.html           # AI governance & ethics content
    ├── foodExp.html          # Food expenditure prediction
    ├── risk.html             # Risk calculator input & results
    ├── google.html           # Google stock enquiry
    ├── apple.html            # Apple stock enquiry
    ├── stock.html            # DBS/UOB agentic AI input
    ├── stockResult.html      # Agentic AI analysis results dashboard
    ├── chatbot.html          # Chatbot menu
    ├── roe.html              # Predefined: Return on Equity explanation
    ├── groqReply.html        # Chatbot general Q&A response
    └── generalQuestion.html  # General question input form
```

---

## Prerequisites

- Python 3.8 or higher
- `pip` package manager
- A [Groq API key](https://console.groq.com/) (free tier available)
- Internet access (required for `yfinance` market data and Groq API calls)

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd an6802
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Full dependency list:

```
flask
gunicorn
scikit-learn
joblib
groq
yfinance
statsmodels
numpy
pandas
```

---

## Configuration

Set your Groq API key as an environment variable before starting the app:

```bash
# macOS/Linux
export GROQ_API_KEY="your_groq_api_key_here"

# Windows (Command Prompt)
set GROQ_API_KEY=your_groq_api_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"
```

> **Security:** Never commit your API key to version control. Add `.env` to your `.gitignore` if using a dotenv file.

---

## Running the App

### Development

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

### Production (Gunicorn)

```bash
gunicorn app:app

# With custom workers and port
gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
```

---

## Usage Guide

### Step 1 — Welcome
Enter your name and submit to reach the main dashboard.

### Step 2 — Navigate from Dashboard

| Module | What it does |
|---|---|
| Ethics Quiz | True/false business ethics scenarios with feedback |
| Economics | Read core economics concepts |
| AI Governance | Learn about responsible AI in finance |
| Food Expenditure | Predict weekly food costs from salary input |
| Risk Calculator | Input prices and receive VaR, Sharpe, drawdown metrics |
| Google / Apple | Open external stock analysis tools in a new tab |
| DBS / UOB Analysis | Trigger full agentic AI stock analysis pipeline |
| Chatbot | Ask financial questions via Groq LLM |

### Ethics Quiz
Select True or False for each scenario. The app routes to a green confirmation page for correct answers or a red explanation page for incorrect ones.

### Food Expenditure Prediction
Enter a numerical salary value. The pre-trained model returns an estimated weekly food expenditure.

### Risk Calculator
Supply historical price data as prompted. The app computes and displays VaR (95%), Sharpe Ratio, max drawdown, and annualised volatility.

### Agentic AI Stock Analysis (DBS / UOB)
Select a stock and submit. The pipeline runs autonomously across five steps and renders a full results dashboard including metric cards, a 5-day ARIMA forecast, sentiment analysis, an AI recommendation, and a step-by-step agent reasoning log.

### AI Chatbot
Choose a predefined question for a concise explanation, or type your own financial question. Responses are plain text without markdown formatting for clean in-browser display.

---

## Agentic AI Workflow

The DBS/UOB analysis module implements a five-step autonomous agent pipeline:

```
Stock Selected
      │
      ▼
Step 1: Fetch Data
  └─ yfinance pulls 6-month OHLCV history
      │
      ▼
Step 2: Risk Metrics
  └─ Annual volatility, Sharpe ratio, 95% VaR, max drawdown, risk band
      │
      ▼
Step 3: ARIMA Forecast
  └─ ARIMA(2,1,2) → 5-day price forecast + 90% confidence interval
      │
      ▼
Step 4: Sentiment Analysis
  └─ Groq LLM analyses recent news → Positive / Neutral / Negative + key factors
      │
      ▼
Step 5: Recommendation
  └─ Groq LLM synthesises all signals → MAS-compliant Buy / Hold / Sell
```

All intermediate steps are logged and displayed in the results dashboard under **Agent Reasoning Steps**.

> **Disclaimer:** All outputs are for educational purposes only and do not constitute financial advice. Investment decisions should be made with guidance from a licensed financial professional, in accordance with MAS regulatory guidelines.

---

## Deployment

The app is deployed on **Render** at [https://an6802-p495.onrender.com](https://an6802-p495.onrender.com).

To deploy your own instance:

- Use **Gunicorn** as the WSGI server (included in `requirements.txt`).
- Set `GROQ_API_KEY` in your hosting platform's environment variables (e.g., Render, Railway, Heroku Config Vars).
- Ensure `foodexp.pkl` is bundled with the deployment — it is not generated at runtime.
- Do not run Flask in `debug=True` mode in production.
- The `yfinance` data fetch and Groq API calls require outbound internet access from your server.

---

## Design System

The UI uses a Claude-inspired aesthetic:

| Token | Value | Usage |
|---|---|---|
| Background | `#F5F0E8` | Warm beige page background |
| Accent | `#D97706` | Buttons, links, focus rings |
| Text | `#3D3929` | Primary body text |
| Muted text | `#78716C` | Paragraphs, descriptions |
| Border | `#D4CFC4` | Input and card borders |
| Font (body) | DM Sans | UI elements and forms |
| Font (headings) | Libre Baskerville | Page and section titles |
