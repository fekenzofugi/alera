from langchain_community.llms.ollama import Ollama


model = Ollama(model="tinyllama", base_url="http://localhost:11434")