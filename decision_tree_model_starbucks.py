from pathlib import Path
import io
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Starbucks Beverage Classifier", page_icon="☕", layout="wide")
st.title("☕ Starbucks Beverage Classifier")
st.caption("Explore Starbucks nutrition data and train a decision-tree model.")

DATA_PATHS = [Path("starbucks.csv"), Path("data/starbucks.csv")]
TARGET_DEFAULT = "Beverage_category"

NUMERIC_FEATURES_PREFERRED = [
    "Calories", "Total Fat (g)", "Trans Fat (g)", "Saturated Fat (g)",
    "Sodium (mg)", "Total Carbohydrates (g)", "Cholesterol (mg)",
    "Dietary Fibre (g)", "Sugars (g)", "Protein (g)",
    "Vitamin A (% DV)", "Vitamin C (% DV)", "Calcium (% DV)",
    "Iron (% DV)", "Caffeine (mg)"
]

def load_csv_bytes(file_bytes):
    """Read a CSV and normalize column names and numeric nutrition fields."""
    frame = pd.read_csv(io.BytesIO(file_bytes))
    frame.columns = frame.columns.astype(str).str.strip()

    # Convert percentage strings and numeric-looking text columns to numbers.
    for column in frame.columns:
        if column in ("Beverage_category", "Beverage", "Beverage_prep"):
            continue
        if frame[column].dtype == "object":
            cleaned = frame[column].astype(str).str.replace("%", "", regex=False).str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().any():
                frame[column] = converted
    return frame

def get_dataset():
    uploaded = st.file_uploader("Optional: upload a replacement Starbucks CSV", type=["csv"])
    if uploaded is not None:
        return load_csv_bytes(uploaded.getvalue()), f"Uploaded file: {uploaded.name}"

    for path in DATA_PATHS:
        if path.exists():
            return load_csv_bytes(path.read_bytes()), f"Loaded repository file: {path.as_posix()}"

    st.error(
        "I couldn't find starbucks.csv. Upload it above, or add starbucks.csv "
        "to the root of your GitHub repository beside app.py."
    )
    st.stop()

df, source_label = get_dataset()
st.success(source_label)

if TARGET_DEFAULT not in df.columns:
    st.error(f"Required target column '{TARGET_DEFAULT}' was not found. Check the CSV column names.")
    st.stop()

st.subheader("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", int(df.isna().sum().sum()))

with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Missing values and descriptive statistics"):
    st.write("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing values"), use_container_width=True)
    st.write("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

st.subheader("2. Visualize the dataset")
categorical_candidates = [
    col for col in df.columns
    if 2 <= df[col].nunique(dropna=True) <= 15
]
if categorical_candidates:
    chart_column = st.selectbox("Choose a column for a category-count chart", categorical_candidates)
    counts = df[chart_column].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution of {chart_column}")
    ax.set_xlabel(chart_column)
    ax.set_ylabel("Number of rows")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

if "Calories" in df.columns:
    st.write("Calories by beverage category")
    calorie_df = df[["Beverage_category", "Calories"]].copy()
    calorie_df["Calories"] = pd.to_numeric(calorie_df["Calories"], errors="coerce")
    calorie_df = calorie_df.dropna()
    if not calorie_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        calorie_df.boxplot(column="Calories", by="Beverage_category", ax=ax, rot=35)
        ax.set_title("Calories by Beverage Category")
        ax.set_xlabel("Beverage category")
        ax.set_ylabel("Calories")
        fig.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.subheader("3. Train and evaluate the decision tree")
available_features = [
    col for col in df.columns
    if col != TARGET_DEFAULT and pd.api.types.is_numeric_dtype(df[col])
]
default_features = [col for col in NUMERIC_FEATURES_PREFERRED if col in available_features]
if not default_features:
    default_features = available_features

if not available_features:
    st.error("No numeric feature columns were found after cleaning the CSV.")
    st.stop()

features = st.multiselect(
    "Choose numeric nutrition features",
    options=available_features,
    default=default_features
)
test_percent = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5)
max_depth = st.slider("Decision tree maximum depth", min_value=2, max_value=12, value=5)

if not features:
    st.warning("Select at least one feature to train the model.")
    st.stop()

model_df = df[features + [TARGET_DEFAULT]].copy()
model_df[TARGET_DEFAULT] = model_df[TARGET_DEFAULT].astype("string").str.strip()
model_df = model_df.dropna(subset=[TARGET_DEFAULT])

for col in features:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    median = model_df[col].median()
    model_df[col] = model_df[col].fillna(0 if pd.isna(median) else median)

# Stratified train/test splitting needs at least two rows per target category.
class_counts = model_df[TARGET_DEFAULT].value_counts()
model_df = model_df[model_df[TARGET_DEFAULT].isin(class_counts[class_counts >= 2].index)]
if model_df[TARGET_DEFAULT].nunique() < 2:
    st.error("The target column needs at least two categories with at least two rows each.")
    st.stop()

if st.button("Train model", type="primary"):
    try:
        X = model_df[features]
        y = model_df[TARGET_DEFAULT].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_percent / 100, random_state=42, stratify=y
        )
        model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        st.metric("Test accuracy", f"{accuracy * 100:.2f}%")
        st.subheader("Classification report")
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        result_df = pd.DataFrame({
            "Beverage": df.loc[y_test.index, "Beverage"].values if "Beverage" in df.columns else y_test.index,
            "Beverage_prep": df.loc[y_test.index, "Beverage_prep"].values if "Beverage_prep" in df.columns else "",
            "Actual_Category": y_test.values,
            "Predicted_Category": predictions,
        })
        st.subheader("Sample predictions")
        st.dataframe(result_df.head(10), use_container_width=True)
        st.download_button(
            "Download prediction results (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="starbucks_predictions.csv",
            mime="text/csv",
        )

        model_bytes = pickle.dumps(model)
        st.download_button(
            "Download trained model (.pkl)",
            data=model_bytes,
            file_name="starbucks_beverage_classifier.pkl",
            mime="application/octet-stream",
        )
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.info("Check the dataset columns and try a different feature selection or test size.")
