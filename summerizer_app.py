# imports
import streamlit as st
import pdfminer
import requests
#from pdfminer import high_level
from pdfminer.layout import LAParams
import pandas as pd
import arxiv
import textwrap
from bs4 import BeautifulSoup
import openai
import streamlit as st
from streamlit_tags import st_tags  
from io import BytesIO


# converts pdf to text
@st.experimental_memo()
def pdf_2_text(file: BytesIO) -> str:
    with open(file, "rb") as fp:
        text = pdfminer.high_level.extract_text(
                    fp, 
                    codec="utf-8",
                    laparams=LAParams(
                        line_margin=0.5,
                        word_margin=0.1,
                        boxes_flow=0.5,
                        detect_vertical=True,
                        all_texts=True,),
                    maxpages=0,)
    return text

# chunks large texts to 4000 tokens
def chunk_text(text):
    return textwrap.wrap(text, 4000)

# texts from the web
def url_2_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    for tag in soup.find_all():
        tag.attrs = {}
    text = soup.get_text()
    return textwrap.fill(text, width=150)

# arxiv pdfs
def search_arxiv(query):
    search = arxiv.Search(
        query = query,
        max_results = 300,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending
        )

    all_data = []
    for result in search.results():
        temp = ["","","","",""]
        temp[0] = result.title
        temp[1] = result.published
        temp[2] = result.entry_id
        temp[3] = result.summary
        temp[4] = result.pdf_url
        all_data.append(temp)

    column_names = ['Title','Date','Id','Summary','URL']
    df = pd.DataFrame(all_data, columns=column_names)

    print("Number of papers extracted : ",df.shape[0])
    return df
@st.experimental_memo()
def arxiv_2_file(url, filename):
    url = url
    filename = filename+'.pdf'
    response = requests.get(url)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        res = "PDF file downloaded successfully."
    else:
        res = "Failed to download the PDF file."
    return res


# summerizer
def get_completion(prompt, model="gpt-3.5-turbo"): # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def summerizer(text):
    prompt = f"""
        Your task is to extract and summerize relevant information from a text extracted from an article/paper by first 
        cleaning the text of irrelevant or unclear information, that is, ignore information that is irrelevant to the 
        purpose of the paper and any unclear part of the text. 
        From the text below, delimited by triple quotes extract the relevant information. Limit to 2000 words. 
        Text: ```{text}```
        """
    return get_completion(prompt)


# WEB PAGE
st.set_page_config(layout="centered", page_title="GPT-Powered Paper and Article Summerizer", page_icon="❄️" )


c1, c2 = st.columns([0.32, 2])
# The snowflake logo will be displayed in the first column, on the left.
with c1:
    st.image("images/logo.jpg", width=85,)

with c2:
    st.caption("")
    st.title("GPT-Powered Paper and Article Summerizer")
# We need to set up session state via st.session_state so that app interactions don't reset the app.
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ SIDEBAR CONTENT ############
st.sidebar.write("")

# For elements to be displayed in the sidebar, we need to add the sidebar element in the widget.
# We create a text input field for users to enter their API key.
API_KEY = st.sidebar.text_input(
    "Enter your OPENAI API key",
    help="Once you created you OPENAI account, you can get your free API token in your settings page: https://platform.openai.com/account/api-keys",
    type="password",
)

openai.api_key = API_KEY 

st.sidebar.markdown("---")
# Let's add some info about the app to the sidebar.
st.sidebar.write(
    """
App created by [Kabir Hamzah](https://twitter.com/kabir_hamzah) using [Streamlit](https://streamlit.io/)🎈 and [ChatGPT](https://chat.openai.com/) model.
""")

ArtTab, LocTab, ArcTab= st.tabs(["Article URL", "Local Paper", "Archive Link"])

with ArtTab:
    url = st.text_input("Provide url to desired article: ")
    if url:
        text = url_2_text(url)
        text = summerizer(text)
        st.text(textwrap.fill(text, 150))

with LocTab:
    file = st.file_uploader("Select PDF file")
    if file:
        doc = pdf_2_text(file.name)
        texts = chunk_text(doc)
        # print(len(texts[0]))
        # print(texts[0])
        all_sum_texts = []
        for text in texts:
            sum_text = summerizer(text)
            all_sum_texts.append(sum_text)
        st.text(textwrap.fill(' '.join(all_sum_texts), 150))


with ArcTab:
    topic = st.text_input("Enter the topic you need to search for: ")
    if topic:
        df = search_arxiv(topic)
        st.dataframe(df, use_container_width=True)
        url_and_name = st.text_input('Enter url and name of desired paper to download',
        help="Copy url from the table above and entire a desired file name, separate the url and name using a comma.")

        if url_and_name:
            url = url_and_name.split(',')[0]
            filename = url_and_name.split(',')[1]
            text = arxiv_2_file(url, filename)
            #text = summerizer(text)
            st.text(text)
