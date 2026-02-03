# Airline Sentiment Analysis & Trend Forecasting

An end-to-end NLP and Time-Series project that performs sentiment analysis on airline tweets using **BERT** and predicts future sentiment trends using **Facebook Prophet**. The project features a modular Python architecture and an interactive **Streamlit** dashboard for real-time analysis.

## Key Features

* **Deep Learning Sentiment Analysis**: Fine-tuned `bert-base-uncased` model for high-accuracy sentiment classification (Positive, Neutral, Negative).
* **Time-Series Forecasting**: Predictive analysis of negative sentiment trends for the upcoming 7 days using Prophet.
* **Interactive Dashboard**: A comprehensive Streamlit interface for real-time tweet analysis and historical trend visualization.
* **Modular Architecture**: Professional project structure separating experimental research (Notebooks) from production-ready code (Src).
* **Interactive Testing**: A dedicated script (`test_interactive.py`) for quick CLI-based model testing.



## Tech Stack

* **Language**: Python
* **Deep Learning**: Transformers (Hugging Face), PyTorch
* **Forecasting**: Facebook Prophet
* **Web Framework**: Streamlit
* **Data Processing**: Pandas, NLTK, Scikit-learn
* **Training Environment**: Google Colab (T4 GPU)


 Model Performance
The BERT model was fine-tuned on the Twitter US Airline Sentiment dataset:

!!!!!!!!!!!!!  YOU CAN DOWNLOAD MY BERT MODEL FROM THIS DRIVE LINK:               
(https://drive.google.com/file/d/1yuH7xcSsjuVy6aU2f73ubTtXLOvdLh7P/view?usp=sharing)

Accuracy: ~81% (Restored to best weights from Epoch 2 to prevent overfitting).

Optimization: Implemented EarlyStoppingCallback with patience=2.

 Installation & Usage
Clone the repository:

* Bash
git clone [https://github.com/ZekiKurt0/Social-Media-Sentiment-Trend.git](https://github.com/ZekiKurt0/social-media-sentiment-trend.git)
cd social-media-sentiment-trend
Install dependencies:

* Bash
pip install -r requirements.txt
Run the Dashboard:

* Bash
$env:PYTHONPATH = "."
streamlit run src/dashboard.py
Try Interactive Test:

* Bash
python test_interactive.py