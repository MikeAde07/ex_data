# AI CSV Agent + RAG 🤖
- ## Upload your csv file and ask questions with A.i agent to recieve further insight about data within the files. Uses langchain, OpenAI, RAG, Streamlit, and fully Dockerized.

## 🪀Features 
- Conversational chat is saved in memory for Agent to pull
- Chroma DB is the vector database used
- OpenAI embeddings
- Retrieval Augmented Generation structured in a function for Agent to utilize
- streamlit front-end
- Agent built using Langchain framework
- Dockerized for local and cloud deployment


## 👨🏾‍💻Tech Stack
- **Frontend**: Streamlit
- **Backend**: Langchain, OpenAI (GPT + Embeddings)
- **RAG**: Chroma DB
- **Environment**: Python 3.12, Docker
- **Tools**: PythonREPLTool, Vector Search, OpenAI API


## Getting Started
1. Clone the Repository:
   git clone https://github.com/MikeAde07/ex_data.git
   cd ex_data
2. Set up environment:
   Create a .env file in the root directory and add your OpenAI API key:
   OPENAI_API_KEY=your_openai_key
3. Build and Run with Docker:
   docker compose up --build
   Once running, visit http://localhost:8000 in your browser

## 📂Folder Structure
- main.py (Streamlit app logic)
- vector.py (Vectorstore & retrieval logic)
- htmlTemplates.py (css templates for agent & user)
- Dockerfile
- docker-compose.yaml
- requirements.txt
- .env
- .dockerignore
- README.md

## 🚀Example Use Cases
- "What are the Top 5 sales in the SW Region?"
- "How many customers were served on Friday?"

## 🔨 Agent Tools Used
| Tool | Description |
|------|-------------|
|  RAG Tool    | Retrieves semantic matches from CSV via Chroma DB             |
| PythonREPLTool      |    Allows agent to execute live python code during conversation or reasoning process    |
|OpenAI LLM| LLM to converse and answer queries|

## 🐳Docker Notes
* All CSV processing and vectorization happen within container
* Vector DB is mounted to ./chroma_db locally for persistence
* You can host this app on Render, Railway, or AWS EC2
