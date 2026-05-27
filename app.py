# app.py
# Advanced Phishing Detection Flask Web App

from flask import Flask, render_template_string, request
import re
import socket
import ssl
import whois
import requests
import pickle
import numpy as np
import tldextract
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from datetime import datetime
import ipaddress

app = Flask(__name__)

# =========================
# LOAD TRAINED MODEL
# =========================

try:
    with open("phishing_model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    model = None

# =========================
# HTML TEMPLATE
# =========================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Website Detector</title>
    <style>
        body{
            background:#0f172a;
            color:white;
            font-family:Arial;
            padding:40px;
        }

        .container{
            max-width:800px;
            margin:auto;
        }

        input{
            width:100%;
            padding:14px;
            border:none;
            border-radius:10px;
            font-size:16px;
        }

        button{
            margin-top:15px;
            padding:12px 25px;
            border:none;
            border-radius:10px;
            background:#2563eb;
            color:white;
            font-size:16px;
            cursor:pointer;
        }

        .result{
            margin-top:30px;
            padding:20px;
            border-radius:12px;
        }

        .safe{
            background:#14532d;
        }

        .danger{
            background:#7f1d1d;
        }

        .warning{
            background:#78350f;
        }

        ul{
            line-height:1.8;
        }
    </style>
</head>
<body>

<div class="container">

    <h1>Advanced Phishing Website Detector</h1>

    <form method="POST">
        <input type="text" name="url" placeholder="Enter URL..." required>
        <button type="submit">Analyze</button>
    </form>

    {% if result %}

        <div class="result {{ result.class }}">

            <h2>{{ result.verdict }}</h2>

            <h3>Risk Score: {{ result.score }}/100</h3>

            <ul>
            {% for item in result.details %}
                <li>{{ item }}</li>
            {% endfor %}
            </ul>

        </div>

    {% endif %}

</div>

</body>
</html>
"""

# =========================
# URL VALIDATION
# =========================

def normalize_url(url):
    if not url.startswith("http"):
        url = "http://" + url
    return url

# =========================
# CHECKS
# =========================

def has_ip(domain):
    try:
        ipaddress.ip_address(domain)
        return True
    except:
        return False

def suspicious_words(url):
    keywords = [
        "login",
        "verify",
        "update",
        "banking",
        "secure",
        "account",
        "paypal",
        "signin",
        "password",
        "confirm",
        "wallet"
    ]

    for word in keywords:
        if word in url.lower():
            return True

    return False

def domain_age(domain):

    try:
        w = whois.whois(domain)

        creation_date = w.creation_date

        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date is None:
            return 0

        age = (datetime.now() - creation_date).days

        return age

    except:
        return 0

def ssl_valid(domain):

    try:
        context = ssl.create_default_context()

        with socket.create_connection((domain, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=domain):
                return True
    except:
        return False

def excessive_subdomains(domain):
    return domain.count('.') > 3

def fetch_html(url):

    try:
        r = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        return r.text

    except:
        return ""

def detect_iframe(html):

    soup = BeautifulSoup(html, "html.parser")

    if soup.find("iframe"):
        return True

    return False

def detect_obfuscated_js(html):

    suspicious_patterns = [
        "eval(",
        "escape(",
        "unescape(",
        "document.write(",
        "fromCharCode"
    ]

    for pattern in suspicious_patterns:
        if pattern in html:
            return True

    return False

# =========================
# FEATURE EXTRACTION
# =========================

def extract_ml_features(url):

    parsed = urlparse(url)

    domain = parsed.netloc

    features = [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        int("https" in url),
        int(has_ip(domain)),
        int(excessive_subdomains(domain)),
        int(suspicious_words(url))
    ]

    return np.array(features).reshape(1, -1)

# =========================
# MAIN ANALYSIS
# =========================

def analyze_url(url):

    url = normalize_url(url)

    parsed = urlparse(url)

    domain = parsed.netloc

    score = 0

    details = []

    # URL Length
    if len(url) > 75:
        score += 15
        details.append("Long URL detected")

    # IP Address
    if has_ip(domain):
        score += 20
        details.append("Uses IP address instead of domain")

    # Suspicious Keywords
    if suspicious_words(url):
        score += 15
        details.append("Suspicious phishing keywords found")

    # Subdomains
    if excessive_subdomains(domain):
        score += 10
        details.append("Too many subdomains")

    # SSL
    if not ssl_valid(domain):
        score += 20
        details.append("Invalid or missing SSL certificate")

    # Domain Age
    age = domain_age(domain)

    if age < 180:
        score += 15
        details.append(f"Very new domain ({age} days old)")

    # HTML Analysis
    html = fetch_html(url)

    if html:

        if detect_iframe(html):
            score += 10
            details.append("Hidden iframe detected")

        if detect_obfuscated_js(html):
            score += 15
            details.append("Obfuscated JavaScript detected")

    # ML Prediction
    if model:

        try:

            features = extract_ml_features(url)

            prediction = model.predict(features)[0]

            if prediction == -1:
                score += 20
                details.append("ML model predicts phishing")

        except:
            pass

    # FINAL RESULT
    if score >= 70:
        verdict = "PHISHING WEBSITE"
        css = "danger"

    elif score >= 40:
        verdict = "SUSPICIOUS WEBSITE"
        css = "warning"

    else:
        verdict = "LEGITIMATE WEBSITE"
        css = "safe"

    return {
        "verdict": verdict,
        "score": score,
        "details": details,
        "class": css
    }

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET", "POST"])

def home():

    result = None

    if request.method == "POST":

        url = request.form["url"]

        result = analyze_url(url)

    return render_template_string(HTML, result=result)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(debug=True)