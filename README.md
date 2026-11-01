# AI Search Assistant

An AI-powered search assistant built using **LangChain, Groq, and Streamlit**. It uses an AI agent to retrieve information from multiple sources, including DuckDuckGo, Wikipedia, and Arxiv.

### Live Demo

[**Try the AI Search Assistant**](YOUR_STREAMLIT_APP_URL)

### Features

- **AI-powered responses:** Uses Groq-hosted LLMs to generate responses.
- **Web Search:** Retrieves information using DuckDuckGo.
- **Wikipedia:** Accesses general knowledge and factual information.
- **Research Papers:** Searches academic papers using Arxiv.
- **Tool-calling Agent:** Automatically selects appropriate tools based on user queries.
- **Interactive Chat:** Maintains conversation history with a clear chat option.

### Tech Stack

- Python
- Streamlit
- LangChain
- Groq API (GPT-OSS 120B)
- DuckDuckGo Search
- Wikipedia API
- Arxiv API

### Run Locally

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd ai-search-assistant
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your Groq API key:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

### Configuration

The application uses a Groq API key stored securely through environment variables or Streamlit Secrets.

### Author

**Bupendra Devegade**