from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
Settings.llm = Ollama(
    model="qwen2.5:7b-instruct-q5_K_M",
    request_timeout=120
)

documents = SimpleDirectoryReader(
    input_dir=".",
    required_exts=[".md"],
    recursive=True
).load_data()

print(f"✅ Loaded {len(documents)} knowledge files")

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./nihaixia_index")
print("✅ Knowledge base built successfully — folder: nihaixia_index/")