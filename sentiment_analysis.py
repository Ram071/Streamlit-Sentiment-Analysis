from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis")


# --------------------------------------------------
# Text Analysis
# --------------------------------------------------

with st.expander("Analyze Text", expanded=True):

    text = st.text_area(
        "Enter text:",
        placeholder="Type your text here..."
    )

    if text:

        blob = TextBlob(text)

        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        st.write(
            "Polarity:",
            round(polarity, 2)
        )

        st.write(
            "Subjectivity:",
            round(subjectivity, 2)
        )

        if polarity > 0:
            st.success("😊 Positive")

        elif polarity < 0:
            st.error("😞 Negative")

        else:
            st.info("😐 Neutral")


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

    pre = st.text_area(
        "Clean Text:",
        placeholder="Enter text to clean..."
    )

    if pre:

        cleaned_text = cleantext.clean(
            pre,
            clean_all=False,
            extra_spaces=True,
            stopwords=True,
            lowercase=True,
            numbers=True,
            punct=True
        )

        st.write("Cleaned Text:")
        st.write(cleaned_text)


# --------------------------------------------------
# CSV / Excel Analysis
# --------------------------------------------------

with st.expander("Analyze CSV / Excel"):

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["csv", "xlsx", "xls"]
    )


    def score(text):
        """Return the sentiment polarity score."""

        blob = TextBlob(str(text))

        return blob.sentiment.polarity


    def analyze(value):
        """Convert polarity score into sentiment."""

        if value > 0:
            return "Positive"

        elif value < 0:
            return "Negative"

        return "Neutral"


    if uploaded_file:

        try:

            # Read CSV
            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(
                    uploaded_file
                )

            # Read Excel
            else:

                df = pd.read_excel(
                    uploaded_file
                )


            # Remove unnecessary index column
            if "Unnamed: 0" in df.columns:

                df.drop(
                    columns=["Unnamed: 0"],
                    inplace=True
                )


            # Check required column
            if "tweets" not in df.columns:

                st.error(
                    "The uploaded file must contain "
                    "a 'tweets' column."
                )

            else:

                # Calculate sentiment score
                df["score"] = df["tweets"].apply(
                    score
                )

                # Calculate sentiment
                df["analysis"] = df["score"].apply(
                    analyze
                )


                # Display results
                st.subheader(
                    "Analysis Results"
                )

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )


                # Download results
                @st.cache_data
                def convert_df(dataframe):

                    return dataframe.to_csv(
                        index=False
                    ).encode("utf-8")


                csv = convert_df(df)


                st.download_button(
                    label="⬇️ Download Results",
                    data=csv,
                    file_name="sentiment.csv",
                    mime="text/csv"
                )


        except Exception as error:

            st.error(
                f"Error processing file: {error}"
            )
