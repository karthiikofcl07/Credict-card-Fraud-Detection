# LinkedIn Post Caption

---

## Option A — Story-first (Recommended)

🚨 **I built an AI system that detects credit card fraud in real time — trained on 1.85 million transactions.**

The results blew me away:

✅ **Random Forest AUC-ROC: 0.9941**
✅ **Fraud F1-Score: 0.9048** (catches 93% of fraud)
✅ **Accuracy: 98.59%** on 555K held-out transactions
✅ Full inference in < 50ms per transaction

Here's what I learned building this end-to-end ML pipeline 👇

**The data challenge:**
Only 0.52% of 1.85M transactions were fraudulent — a brutal class imbalance that makes naive models useless. The fix? `class_weight='balanced'` + evaluating on AUC-ROC and F1 instead of accuracy.

**The surprising features:**
Transaction *amount* and *geographic distance* (card-holder ↔ merchant) turned out to be the top predictors — more powerful than time of day or merchant category. Fraud transactions average $527 vs just $67 for legitimate ones.

**The stack:**
→ Python · pandas · scikit-learn · matplotlib · seaborn
→ Flask REST API with responsive real-time dashboard
→ Jupyter notebook with full EDA and annotated analysis

📊 The project includes:
• 9 publication-quality visualisations
• 3 algorithms benchmarked head-to-head
• A live web app where you can test any transaction
• Clean, documented code ready for production

🔗 GitHub repo in the comments below!

#MachineLearning #DataScience #Python #FraudDetection #AI #sklearn #Flask #FinTech #MLOps #OpenToWork

---

## Option B — Metrics-first (Short & punchy)

🛡️ **Just shipped: Credit Card Fraud Detection AI**

📊 **1,851,394 transactions** analysed
🌲 **Random Forest AUC = 0.9941**
⚡ **Real-time predictions** via Flask API
🐍 **Full open-source** on GitHub

Built with scikit-learn · pandas · matplotlib · Flask

The trickiest part? The dataset is only 0.52% fraud. Solving that imbalance problem was more important than model selection.

Drop a comment if you want to see how I handled it 👇

GitHub link in bio!

#Python #MachineLearning #DataScience #AI #FraudDetection #Flask #FinTech

---

## Option C — Educational (Max engagement)

💳 **Why do banks decline legitimate transactions? I built the ML system that explains it.**

I trained a fraud detection model on 1.85M credit card transactions. Here are 5 surprising things I found:

**1️⃣ Amount is everything**
Fraud transactions average $527. Legitimate ones average $67. A single feature explains ~28% of the model's decisions.

**2️⃣ 3 AM is the riskiest time**
Fraud rate spikes 3–4× during late-night hours. Your bank's algorithm is watching the clock.

**3️⃣ "misc_net" is a red flag**
Online miscellaneous purchases have a fraud rate 10× higher than grocery stores. Card-Not-Present fraud is real.

**4️⃣ Geographic distance matters**
If your card is used 500 km from home at 2 AM, the model notices. Every time.

**5️⃣ 99% accuracy is meaningless here**
With only 0.52% fraud in the dataset, a model that predicts "NEVER FRAUD" gets 99.5% accuracy and catches nothing. That's why we use AUC-ROC and F1 instead.

My Random Forest scored **AUC 0.9941** and catches **93% of fraud** with **88% precision**.

Full code + interactive demo on GitHub (link in comments) 🔗

#MachineLearning #DataScience #AI #FraudDetection #Python #FinTech #DataAnalytics #sklearn
