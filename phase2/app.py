import os
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector | AI Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Clean, Modern Dashboard Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 1.2rem 0;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #7928CA 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1rem;
    }
    
    /* Result card styling */
    .result-card {
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    .result-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.04) 100%);
        border-left: 6px solid #EF4444;
    }
    .result-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.04) 100%);
        border-left: 6px solid #10B981;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .badge-spam {
        background-color: #EF4444;
        color: white;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-ham {
        background-color: #10B981;
        color: white;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    
    /* Stat cards */
    .metric-box {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Pretrained Model Pipeline
# ---------------------------------------------------------
MODEL_FILE = os.path.join(os.path.dirname(__file__), "spamcollection.joblib")

@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_FILE):
        return None, f"Model file '{MODEL_FILE}' not found. Please ensure 'spamcollection.joblib' is in the project folder."
    try:
        model = joblib.load(MODEL_FILE)
        return model, None
    except Exception as e:
        return None, str(e)

model, load_err = load_classifier()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/spam.png", width=105)
    st.title("🛡️ Spam Guard AI")
    st.caption("AI-Driven Text Classification")
    
    if model is not None:
        st.success("✅ Model Pipeline Loaded", icon="🟢")
    else:
        st.error(f"❌ Failed to load model: {load_err}", icon="🔴")
        
    st.markdown("---")
    st.markdown("### 📊 Model Specifications")
    st.markdown("""
    - **Pipeline:** `TF-IDF + MultinomialNB`
    - **Feature Extractor:** `TfidfVectorizer`
    - **Classification Model:** `Multinomial Naive Bayes`
    - **Benchmark Accuracy:** `95.16%`
    - **Spam Precision:** `100%` (0 False Positives)
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Guide (मार्गदर्शिका)")
    st.markdown("""
    1. **Single Check:** Type any SMS / email message or select one from presets.
    2. Click **Classify** to detect Spam vs Ham.
    3. View risk score and detected spam triggers.
    4. **Bulk Check:** Upload your CSV file to process multiple SMS messages at once!
    """)
    
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-Learn")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ SMS & Message Spam Detector</div>
    <div class="sub-title">Real-time Machine Learning detection for spam, lottery scams, and phishing messages</div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ Model could not be loaded: {load_err}")
    st.stop()

# Helper for predictions
def predict_message(text):
    pred_raw = model.predict([text])[0]
    pred = str(pred_raw).lower()
    classes = [str(c).lower() for c in list(model.classes_)]
    probas = model.predict_proba([text])[0]
    
    prob_dict = {classes[idx]: float(probas[idx]) for idx in range(len(classes))}
    spam_prob = prob_dict.get("spam", 0.0)
    ham_prob = prob_dict.get("ham", 0.0)
    
    return pred, spam_prob, ham_prob

# Common spam keywords to highlight
SPAM_KEYWORDS = [
    "free", "win", "winner", "cash", "prize", "claim", "urgent", "reward", 
    "congratulations", "text win", "entry", "call now", "guaranteed", "selected",
    "offer", "limited time", "click here", "subscribe", "bonus", "won", "won cash",
    "claim now", "account suspended", "verify", "100%", "lottery"
]

def find_keywords(text):
    text_lower = text.lower()
    found = set()
    for kw in SPAM_KEYWORDS:
        if kw in text_lower:
            found.add(kw)
    return list(found)

# ---------------------------------------------------------
# Main Tabs: Single Message & Batch Upload
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["💬 Single Message Classifier", "📁 Bulk / CSV Classifier", "📈 Model Analytics"])

# =========================================================
# TAB 1: Single Message Classifier
# =========================================================
with tab1:
    col_input, col_samples = st.columns([2, 1], gap="medium")
    
    sample_options = {
        "Select an example...": "",
        "🎁 Prize / Lottery (Spam)": "Congratulations! You won a free prize. Click here to claim.",
        "💰 Urgent Cash (Spam)": "URGENT! You have won cash. Call now to collect your reward.",
        "🏆 Free Competition (Spam)": "FREE entry in a weekly competition. Text WIN now to 88088!",
        "👋 Casual Friend (Ham)": "Hi, are we meeting tomorrow for lunch?",
        "🏠 Family Check-in (Ham)": "I will reach home in ten minutes, keep dinner ready.",
        "📚 College Request (Ham)": "Please bring my notebook to college tomorrow morning."
    }
    
    # Session state handling
    if "sms_text" not in st.session_state:
        st.session_state["sms_text"] = ""

    def on_sample_change():
        choice = st.session_state.get("sample_dropdown")
        if choice and choice != "Select an example...":
            st.session_state["sms_text"] = sample_options[choice]

    with col_samples:
        st.markdown("#### 💡 Quick Samples")
        st.selectbox(
            "Test with common scenarios:",
            options=list(sample_options.keys()),
            key="sample_dropdown",
            on_change=on_sample_change
        )
        
        st.markdown("""
        <div style="background:#f1f5f9; padding: 14px; border-radius: 10px; font-size: 0.88rem; color: #475569; line-height: 1.5; border: 1px solid #e2e8f0;">
            <b>💡 Tip:</b> Pick an example above to auto-fill the text area, or directly paste any SMS, WhatsApp, or Email message into the box on the left.
        </div>
        """, unsafe_allow_html=True)
    
    with col_input:
        st.markdown("#### 📝 Message Input")
        user_message = st.text_area(
            "Enter message text below:",
            key="sms_text",
            placeholder="Type or paste message here (e.g., 'Congratulations! You won a free prize...')...",
            height=140
        )
        
        char_count = len(user_message)
        word_count = len(user_message.split()) if user_message.strip() else 0
        st.caption(f"📊 Words: **{word_count}** | Characters: **{char_count}**")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            classify_btn = st.button("🚀 Classify", type="primary", use_container_width=True)
        with col_btn2:
            def clear_text():
                st.session_state["sms_text"] = ""
                st.session_state["sample_dropdown"] = "Select an example..."
            st.button("🧹 Clear", on_click=clear_text)

    # Classification Output
    if classify_btn or (user_message.strip() and st.session_state.get("sample_dropdown") != "Select an example..."):
        if not user_message.strip():
            st.warning("⚠️ Please enter a message to classify.")
        else:
            with st.spinner("Analyzing text patterns..."):
                pred, spam_prob, ham_prob = predict_message(user_message)
                keywords = find_keywords(user_message)
            
            is_spam = (pred == "spam")
            
            st.markdown("---")
            
            if is_spam:
                st.markdown(f"""
                <div class="result-card result-spam">
                    <div class="result-title" style="color: #DC2626;">
                        <span>🚨 SPAM DETECTED</span>
                        <span class="badge-spam">SUSPICIOUS</span>
                    </div>
                    <p style="margin-top: 10px; color: #7F1D1D; font-size: 1.05rem; line-height: 1.5;">
                        This message has been classified as <b>SPAM</b>. It exhibits patterns common in promotional spam, scam prizes, or phishing. Do not click unknown links or share confidential information.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-ham">
                    <div class="result-title" style="color: #059669;">
                        <span>✅ LEGITIMATE (HAM)</span>
                        <span class="badge-ham">SAFE</span>
                    </div>
                    <p style="margin-top: 10px; color: #064E3B; font-size: 1.05rem; line-height: 1.5;">
                        This message is classified as <b>HAM (NOT SPAM)</b>. It appears safe and resembles normal conversational or transactional communication.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence & Probabilities Display
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Prediction Result</div>
                    <div class="metric-value" style="color: {'#DC2626' if is_spam else '#059669'};">
                        {pred.upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Spam Probability</div>
                    <div class="metric-value" style="color: #DC2626;">
                        {spam_prob * 100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Safe (Ham) Probability</div>
                    <div class="metric-value" style="color: #059669;">
                        {ham_prob * 100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"**Spam Risk Meter:** ({spam_prob*100:.1f}% risk score)")
            st.progress(float(spam_prob))
            
            if keywords:
                st.warning(f"🚩 **Detected Suspicious Keywords:** {', '.join([f'**`{k}`**' for k in keywords])}")
            elif is_spam:
                st.info("ℹ️ Linguistic structure and character sequences match spam training signatures.")

# =========================================================
# TAB 2: Bulk / CSV Classifier
# =========================================================
with tab2:
    st.subheader("📂 Batch Process Multiple Messages")
    st.write("Upload a CSV file with SMS messages to evaluate and classify large batches quickly.")
    
    uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, encoding="latin-1")
            st.write(f"📁 Loaded **{len(df_upload)}** rows.")
            st.dataframe(df_upload.head(3), use_container_width=True)
            
            text_col = st.selectbox("Select message text column:", df_upload.columns)
            
            if st.button("⚡ Run Batch Classification", type="primary"):
                with st.spinner("Classifying messages..."):
                    preds = []
                    spam_probs = []
                    ham_probs = []
                    
                    for msg in df_upload[text_col].astype(str):
                        p, sp, hp = predict_message(msg)
                        preds.append(p)
                        spam_probs.append(round(sp * 100, 2))
                        ham_probs.append(round(hp * 100, 2))
                        
                    df_upload["Prediction"] = preds
                    df_upload["Spam_Risk_%"] = spam_probs
                    df_upload["Ham_Confidence_%"] = ham_probs
                    
                    spam_count = (df_upload["Prediction"] == "spam").sum()
                    ham_count = (df_upload["Prediction"] == "ham").sum()
                    total = len(df_upload)
                    
                    st.success("✅ Batch classification completed successfully!")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Messages Processed", total)
                    m2.metric("Detected Spam", f"{spam_count} ({(spam_count/total)*100:.1f}%)")
                    m3.metric("Detected Ham (Safe)", f"{ham_count} ({(ham_count/total)*100:.1f}%)")
                    
                    st.dataframe(df_upload, use_container_width=True)
                    
                    csv_data = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Annotated CSV",
                        data=csv_data,
                        file_name="spam_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
        except Exception as err:
            st.error(f"Error parsing CSV: {err}")
    else:
        st.info("💡 You can upload any CSV file (like `spam.csv` or an export from your application) to classify in bulk.")

# =========================================================
# TAB 3: Model Analytics
# =========================================================
with tab3:
    st.subheader("📊 Model Architecture & Training Summary")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        #### 🧪 Pipeline Components
        - **TF-IDF Vectorizer (`TfidfVectorizer`):**
          Transforms raw SMS text into normalized numerical vectors based on term frequency and document rarity.
        - **Multinomial Naive Bayes (`MultinomialNB`):**
          Computes posterior probabilities P(Spam | Words) using Bayes' theorem with Laplace smoothing.
        - **Preprocessing:** Duplicate removal, punctuation and stopword handling via TF-IDF n-grams.
        """)
        
    with col_info2:
        st.markdown("""
        #### 🎯 Validation Results (Test Set)
        - **Overall Test Accuracy:** `95.16%`
        - **Precision (Ham):** `95%` | **Recall (Ham):** `100%`
        - **Precision (Spam):** `100%` | **Recall (Spam):** `62%`
        - **Zero False-Positive Rate:** When the model tags a message as spam, it is 100% certain, ensuring safe messages never get blocked.
        """)

    st.markdown("---")
    st.subheader("🔍 Test Set Confusion Matrix")
    cm_df = pd.DataFrame(
        [[903, 0], [50, 81]],
        columns=["Predicted Ham (Safe)", "Predicted Spam"],
        index=["Actual Ham (Safe)", "Actual Spam"]
    )
    st.table(cm_df)
