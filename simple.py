from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Initialize Ollama LLM (make sure Ollama is running and model is pulled)
llm = OllamaLLM(model="llama-3.2-3b-it:latest")

# Define a prompt template for summarization
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in a concise paragraph:\n\n{text}"
)

# Example text to summarize
text = """
LangChain is a framework for developing applications powered by language models. 
It enables applications that are context-aware, reasoned, and data-connected. 
LangChain makes it easy to chain together different components to build complex LLM-powered apps.
"""

# Use the new chaining syntax
chain = prompt | llm
summary = chain.invoke({"text": text})

print("Summary:", summary)