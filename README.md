# 💊 PillPal – Your Smart Medicine Companion

PillPal is an AI-powered assistant that helps you **identify medicines**, **find cheaper generic alternatives**, and **stay updated on banned or recalled drugs**.  
Using **image recognition**, **OCR**, and **real-time drug databases**, PillPal makes medicine verification fast, easy, and reliable.

---

## 🚀 Features
- **Visual Drug Recognition** – Identify pills from photos using a deep learning model.
- **OCR Medicine Strip Scanner** – Extract drug name & dosage from medicine strips.
- **Generic Price Suggestions** – Suggest cheaper alternatives from trusted sources (1mg, NetMeds, PharmEasy APIs).
- **Recall & Ban Alerts** – Cross-check with CDSCO & WHO alerts to warn users about unsafe medicines.
- **India-Friendly** – Designed with Indian drug data & APIs in mind.

---

## 🛠 Tech Stack
- **Backend AI**: Python, TensorFlow / PyTorch
- **OCR**: PaddleOCR / Tesseract
- **Scraping / APIs**: BeautifulSoup, Requests, Selenium (if needed)
- **Web App**: Streamlit / Flask
- **Data**:
  - [Pill Image Recognition Dataset (FDA)](https://www.kaggle.com/datasets/omartinsky/fda-approved-drug-pictures)
  - Indian drug names scraped from public sources