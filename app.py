import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models, \
    pull as classification_pull, save_model as classification_save_model, tune_model as classification_tune_model
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, \
    pull as regression_pull, save_model as regression_save_model, tune_model as regression_tune_model

from datetime import datetime
import os

st.set_page_config(layout="wide")


# Initialize an empty DataFrame
df = pd.DataFrame()

# Checking if sourcedata.csv exists
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None, encoding='latin1')

# Initialize model_filename as a global variable
model_filename = None

# Main Content
st.header("AutoStreamML", divider='rainbow')

with st.sidebar:
    st.image('new_automl.png')
    st.title(":red[AutoStreamML] :rocket:")
    choice = st.radio("Navigation", ["Bring in Data", "Profiling", "Machine Learning", "Download"])
    st.info("This is an application that allows you to build an automated ML pipeline using Streamlit")

# Rest of your original code for different sections
if choice == "Bring in Data":
    file = st.file_uploader(":red[Upload] :orange[your] :green[Dataset] :blue[here] :computer:")
    if file:
        df = pd.read_csv(file, index_col=None, encoding='latin1')

        numerical_cols = df.select_dtypes(include=['number']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

        df.to_csv("sourcedata.csv", index=None, encoding='latin1')
        st.dataframe(df)
        st.balloons()

elif choice == "Profiling":
    if not df.empty:  # Check if data is available
        st.title(":blue[Exploratory] :green[Data] :orange[Analysis] :computer:")
        if st.button("Generate Profile Report"):
            try:
                profile_report = ProfileReport(df)
                st_profile_report(profile_report)
            except Exception as e:
                st.error(f"Error generating profile report: {str(e)}")
            profile_report = ProfileReport(df)
            st_profile_report(profile_report)
        st.balloons()
    else:
        st.warning("Please upload data first to generate the profile report.")

elif choice == "Machine Learning":
    if not df.empty:  # Check if data is available
        st.title(":blue[Machine] :green[Learning] :orange[Training] :trident:")
        target = st.selectbox("Select your target column", df.columns)

        # Add radio button for task selection
        task = st.radio("Select ML Task", ["Regression", "Classification"])
        best_model = None

        if st.button("Train model", type="primary"):
            # Display ML experiment settings
            if task == "Regression":
                regression_setup_df = regression_setup(df, target=target, verbose=True, remove_outliers=True)
                st.info("This is the ML experiment settings")
                st.dataframe(regression_pull())

                best_model = regression_compare_models()
            elif task == "Classification":
                classification_setup_df = classification_setup(df, target=target, verbose=True, remove_outliers=True)
                st.info("This is the ML experiment settings")
                st.dataframe(classification_pull())

                best_model = classification_compare_models()

            # Display ML model and scores
            st.info("This is the ML model")
            compare_df = regression_pull() if task == "Regression" else classification_pull()
            st.dataframe(compare_df)

            tuned_best_model = regression_tune_model(best_model) if task == "Regression" else classification_tune_model(
                best_model)

            # Add timestamp to the model filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_filename = f'best_tuned_{task.lower()}_model_{timestamp}.pkl'

            regression_save_model(tuned_best_model,
                                  model_filename) if task == "Regression" else classification_save_model(
                tuned_best_model, model_filename)
        st.balloons()

    else:
        st.warning("Please upload data first to perform machine learning actions.")

elif choice == "Download":
    model_filename = "best_tuned_model.pkl"  # Assuming the model filename

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            st.download_button("Download the model", f.read(), file_name=model_filename)
        st.balloons()
    else:
        st.warning("Please train a model first before attempting to download.")
