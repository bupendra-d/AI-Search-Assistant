import os
import streamlit as st
import wikipedia
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

from langchain_core.tools import Tool

from langchain_classic.agents import (
    create_tool_calling_agent,
    AgentExecutor,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# 1. Configuration
st.set_page_config(page_title="AI Search Assistant")
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

wikipedia.set_user_agent("AISearchAssistant/1.0")


# 2. Search tools
ddg = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(
        backend="lite",
        max_results=5,
    )
)

def safe_search(query):
    try:
        return ddg.run(query)
    except Exception:
        return "DuckDuckGo search is temporarily unavailable. Try using other available sources."


tools = [
    Tool(
        name="Search",
        func=safe_search,
        description="Search the web for current information and general topics."
    ),
    WikipediaQueryRun(
        name="Wikipedia",
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=1000,
        )
    ),
    ArxivQueryRun(
        name="Arxiv",
        api_wrapper=ArxivAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=1000,
        )
    ),
]


# 3. Streamlit interface
st.title("AI Search Assistant")
st.caption("An intelligent search assistant powered by LangChain and Groq.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
    st.rerun()


# 4. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 5. Process user queries
if question := st.chat_input("Ask anything..."):

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    if not api_key:
        st.error("Groq API key not configured. Please check your environment variables.")
        st.stop()

    try:
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=api_key,
            model="openai/gpt-oss-120b",
            temperature=0,
        )

        # Agent prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful AI search assistant.
                
                Available tools:
                - Search: Search the web.
                - Wikipedia: Retrieve general knowledge.
                - Arxiv: Find academic research papers.
                
                Use tools whenever external information is needed.
                Answer general questions directly when possible.
                Never invent search results or sources.
                Provide clear, accurate, and concise answers."""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=5,
            verbose=False,
        )

        # Convert chat history
        history = [
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages[:-1]
        ]

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                result = executor.invoke({
                    "input": question,
                    "chat_history": history,
                })

            answer = result["output"]
            st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        st.error(f"Something went wrong: {type(e).__name__}: {e}")
