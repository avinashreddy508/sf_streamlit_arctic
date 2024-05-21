import streamlit as st
import configparser
from snowflake.snowpark import Session
from snowflake.connector.errors import ProgrammingError
import pandas as pd
import base64
import os 


st.set_page_config(
    page_title="MindEase App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a header. This is an *MindEase* cool app!"
    }
)

pd.set_option("max_colwidth", None)

def intro():

    st.image("2.png", width=150)

    st.title("# Welcome to MindEase! 👋")

    #st.subheader("Prevalence:Mental health disorders are prevalent in the United States, with an estimated 1 in 5 adults experiencing a mental illness each year.",divider='rainbow')

    st.subheader("Prevalence:")
    st.markdown("Mental health disorders are prevalent in the United States, with an estimated 1 in 5 adults experiencing a mental illness each year.")
    
    st.subheader("Specific Disorders:")
    st.markdown("Anxiety disorders are the most common mental illness in the U.S., affecting 40 million adults aged 18 and older, or about 18.1% of the population every year. Major depressive disorder affects approximately 17.3 million adults, or about 7.1% of the U.S. population.  Bipolar disorder affects approximately 4.4% of adults in the U.S. at some point in their lives.Schizophrenia affects about 1.1% of the U.S. adult population.")
    
    st.subheader("Children and Adolescents:")
    st.markdown("Approximately 7.7% of children aged 3-17 years (about 4.5 million) have diagnosed anxiety, while 3.2% (about 1.9 million) have diagnosed depression.Half of all lifetime cases of mental illness begin by age 14, and 75% by age 24.")
    
    st.subheader("Treatment Gap:")
    st.markdown("Despite the high prevalence of mental health conditions, nearly 60% of adults and nearly 50% of children aged 6-17 with a mental illness did not receive mental health services in the previous year.Cost, lack of access to care, stigma, and shortage of mental health professionals are significant barriers to accessing treatment.")
    
    st.subheader("Substance Abuse:")
    st.markdown("Mental health disorders often co-occur with substance abuse disorders. In 2019, 9.5% of adults (aged 18 and older) had a substance use disorder (SUD) in the past year, including 14.5 million adults with an alcohol use disorder and 8.8 million with an illicit drug use disorder.")
    
    st.subheader("Suicide:")
    st.markdown("Suicide is a leading cause of death in the United States. In 2019, there were 47,511 recorded suicides, making it the 10th leading cause of death overall.Suicide rates are highest among American Indian and Alaska Native populations, followed by white populations.")
    
    st.markdown("These statistics highlight the significant impact of mental health disorders in the United States and the importance of increasing access to mental health services, reducing stigma, and promoting mental wellness across all age groups")


### Default Values
num_chunks = 4  # Num-chunks provided as context. Play with this to check how it affects your accuracy
slide_window = 7  # how many last conversations to remember. This is the slide window.

# Snowflake connection parameters
config = configparser.ConfigParser()
config.read('properties.ini')

snowflake_config = config['Snowflake']

user = snowflake_config.get('user')
password = snowflake_config.get('password')
account = snowflake_config.get('account')
role = snowflake_config.get('role')
warehouse = snowflake_config.get('warehouse')
database = snowflake_config.get('database')
schema = snowflake_config.get('schema')

connection_params = {
    'account': account,
    'user': user,
    'password': password,
    'role': role,
    'warehouse': warehouse,
    'database': database,
    'schema': schema
}


session = Session.builder.configs(connection_params).create()
#st.write("Connected to Snowflake")


### Functions

def initialize_session_state():
    if "clear_conversation" not in st.session_state:
        st.session_state.clear_conversation = False
    if "model_name" not in st.session_state:
        st.session_state.model_name = 'mistral-7b'
    if "use_chat_history" not in st.session_state:
        st.session_state.use_chat_history = True
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    initialize_session_state()
    
    #st.title(f":speech_balloon: Chat Document Assistant with Snowflake Cortex")
    #st.write("This is the list of documents you already have and that will be used to answer your questions:")

    st.image("2.png", width=150)
    st.title(f":speech_balloon: Introducing MindEase, your 24/7 mental health companion Document Assistant with Snowflake Cortex")
    st.write("This is the list of documents you already have and that will be used to answer your questions:")
    
    docs_available = session.sql("ls @docs").collect()
    list_docs = [doc["name"] for doc in docs_available]
    st.dataframe(list_docs)

    config_options()
    init_messages()
     
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input("What do you want to know about your products?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
    
            question = question.replace("'", "")
    
            with st.spinner(f"{st.session_state.model_name} thinking..."):
                response = complete(question)
                res_text = response[0].RESPONSE     
            
                res_text = res_text.replace("'", "")
                message_placeholder.markdown(res_text)
        
        st.session_state.messages.append({"role": "assistant", "content": res_text})


def config_options():
    st.sidebar.selectbox('Select your model:', (
        'mixtral-8x7b',
        'snowflake-arctic',
        'mistral-large',
        'llama3-8b',
        'llama3-70b',
        'reka-flash',
        'mistral-7b',
        'llama2-70b-chat',
        'gemma-7b'), key="model_name")
                                           
    # For educational purposes. Users can check the difference when using memory or not
    st.sidebar.checkbox('Do you want that I remember the chat history?', key="use_chat_history", value=True)
    st.sidebar.checkbox('Debug: Click to see summary generated of previous conversation', key="debug", value=True)
    st.sidebar.button("Start Over", on_click=clear_conversation)
    st.sidebar.expander("Session State").write(st.session_state)


def init_messages():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_conversation():
    st.session_state.messages = []


def get_similar_chunks(question):
    cmd = """
        with results as
        (SELECT RELATIVE_PATH,
           VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec,
                    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)) as similarity,
           chunk
        from docs_chunks_table
        order by similarity desc
        limit ?)
        select chunk, relative_path from results 
    """
    
    df_chunks = session.sql(cmd, params=[question, num_chunks]).to_pandas()       

    df_chunks_length = len(df_chunks) - 1

    similar_chunks = ""
    for i in range(0, df_chunks_length):
        similar_chunks += df_chunks._get_value(i, 'CHUNK')

    similar_chunks = similar_chunks.replace("'", "")
             
    return similar_chunks


def get_chat_history():
    # Get the history from the st.session_state.messages according to the slide window parameter
    chat_history = []
    
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range(start_index, len(st.session_state.messages) - 1):
         chat_history.append(st.session_state.messages[i])

    return chat_history


def summarize_question_with_history(chat_history, question):
    # To get the right context, use the LLM to first summarize the previous conversation
    # This will be used to get embeddings and find similar chunks in the docs for context

    prompt = f"""
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natural language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
    
    cmd = """
            select snowflake.cortex.complete(?, ?) as response
          """
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    summary = df_response[0].RESPONSE     

    if st.session_state.debug:
        st.sidebar.text("Summary to be used to find similar chunks in the docs:")
        st.sidebar.caption(summary)

    summary = summary.replace("'", "")

    return summary


def create_prompt(myquestion):
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()

        if chat_history: # There is chat_history, so not the first question
            question_summary = summarize_question_with_history(chat_history, myquestion)
            prompt_context = get_similar_chunks(question_summary)
        else:
            prompt_context = get_similar_chunks(myquestion) # First question when using history
    else:
        prompt_context = get_similar_chunks(myquestion)
        chat_history = ""
  
    prompt = f"""
           You are an expert chat assistant that extracts information from the CONTEXT provided
           between <context> and </context> tags.
           You offer a chat experience considering the information included in the CHAT HISTORY
           provided between <chat_history> and </chat_history> tags.
           When answering the question contained between <question> and </question> tags,
           be concise and do not hallucinate. 
           If you don’t have the information just say so.
           
           Do not mention the CONTEXT used in your answer.
           Do not mention the CHAT HISTORY used in your answer.
           
           <chat_history>
           {chat_history}
           </chat_history>
           <context>          
           {prompt_context}
           </context>
           <question>  
           {myquestion}
           </question>
           Answer: 
           """

    return prompt


def complete(myquestion):
    prompt = create_prompt(myquestion)
    cmd = """
            select snowflake.cortex.complete(?, ?) as response
          """
    
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    return df_response

page_names_to_funcs = {
    "Home": intro,
    "Chat": main
}

demo_name = st.sidebar.selectbox("Navigate MindEase", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


#if __name__ == "__main__":
#    main()
