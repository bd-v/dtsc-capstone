import streamlit as st
import pandas as pd
from functions import main
import os
import io

def display_results(results):
    st.header("Top 10 Words in each Variation")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Words")
        st.text("Size: " + str(results["size_original"]))
        st.dataframe(results["top_original"])
        csv_original = results["original_words"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Original Words as CSV",
            data=csv_original,
            file_name="original_words.csv",
            mime="text/csv"
        )
        
        # st.subheader("Lemmatized Words")
        # st.text("Size: " + str(results["size_lemmatized"]))
        # st.dataframe(results["top_lemmatized"])

        st.subheader("95% SWL (No Stopwords)")
        st.text("Size: " + str(results["size_swl_no_stopwords"]))
        st.dataframe(results["top_swl_no_stopwords"])
        csv_swl_no_stop = results["swl"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download 95% SWL (No Stopwords) as CSV",
            data=csv_swl_no_stop,
            file_name="word_list.csv",
            mime="text/csv"
        )
        

    with col2:
        st.subheader("95% SWL")
        st.text("Size: " + str(results["size_swl"]))
        st.dataframe(results["top_swl"])
        csv_swl = results["swl_no_stopwords"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download 95% SWL as CSV",
            data=csv_swl,
            file_name="word_list_nostopwords.csv",
            mime="text/csv"
        )

        # st.subheader("95% SWL (No Proper Nouns)")
        # st.text("Size: " + str(results["size_swl_no_proper_nouns"]))
        # st.dataframe(results["top_swl_no_proper_nouns"])

# Streamlit app title
st.title("From Data to Dialogue: Unlocking Language for All")

st.markdown("""
    This app allows you to use any text to generate a specialized word list.
    You can also download the word lists as CSV files for further analysis or use in other applications.
""")

# Add two tiles for user choice
st.subheader("Choose Input Method")
option = st.radio("Select one:", ("Upload a File", "Paste Your Text", "Our Choices for You"))

if option == "Upload a File":
    # File uploader
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the file using main.py
        results = main(file_path)

        display_results(results)

        # Clean up the temporary file
        os.remove(file_path)

elif option == "Paste Your Text":
    # Add a text area for user input
    st.subheader("Paste Your Text Below")
    user_text = st.text_area("Paste your text here:")

    if user_text:
        # Save the user input as a temporary text file
        file_path = "temp_user_input.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(user_text)

        # Process the file using main.py
        results = main(file_path)

        display_results(results)

        # Clean up the temporary file
        os.remove(file_path)

elif option == "Our Choices for You":
    st.subheader("Select a Preloaded Text File")
    preloaded_files = ["Alice In Wonderland", "Lord of the Rings - Chapter One", "Titanic"] 
    selected_file = st.selectbox("Select a file:", preloaded_files)
    if selected_file:
        file_path = "data/" + selected_file + ".txt"
        results = main(file_path)
        
        display_results(results)