import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env na pasta raiz
load_dotenv('../.env')

# Configura a chave de API para as bibliotecas do Google
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Variável de ambiente GEMINI_API_KEY não encontrada.")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# --- Carrega a base de conhecimento na inicialização ---
vector_store = None
llm = None # Vamos inicializar o modelo de linguagem aqui

def initialize_knowledge_base():
    global vector_store, llm
    try:
        print("[Retriever Service] Iniciando indexação e carregando LLM...")
        
        # Inicializa o LLM que será usado para gerar as buscas
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)

        knowledge_base_path = '../knowledge_base/'
        
        loader = DirectoryLoader(
            knowledge_base_path,
            glob="**/[!.]*",
            use_multithreading=True,
            loader_cls=lambda path: PyPDFLoader(path) if path.lower().endswith('.pdf') else TextLoader(path, encoding='utf-8')
        )
        docs = loader.load()

        if not docs:
            print("[Retriever Service] Nenhum documento encontrado na pasta knowledge_base.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        vector_store = FAISS.from_documents(splits, embeddings)
        print(f"[Retriever Service] Indexação concluída. {len(splits)} chunks carregados.")

    except Exception as e:
        print(f"[Retriever Service] ERRO ao inicializar a base de conhecimento: {e}")

# --- Define o endpoint de busca ---
@app.route('/buscar', methods=['POST'])
def buscar_contexto():
    global vector_store, llm
    if not vector_store or not llm:
        return jsonify({"error": "A base de conhecimento ou o LLM não estão disponíveis."}), 503

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "A 'query' (pergunta) é obrigatória no corpo da requisição."}), 400

    query = data['query']
    print(f"[Retriever Service] Recebida a query: {query}")

    try:
        # --- INÍCIO DA BUSCA INTELIGENTE ---
        # Usando o MultiQueryRetriever para gerar múltiplas buscas
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(), 
            llm=llm
        )
        docs = retriever.invoke(query)
        # --- FIM DA BUSCA INTELIGENTE ---
        
        context_list = []
        for doc in docs:
            source_name = os.path.basename(doc.metadata.get('source', 'N/A'))
            context_list.append(f"Nome do Arquivo Fonte: {source_name}\nConteúdo: {doc.page_content}")

        context = "\n---\n".join(context_list)
        
        return jsonify({"context": context})

    except Exception as e:
        print(f"[Retriever Service] ERRO durante a busca: {e}")
        return jsonify({"error": "Ocorreu um erro interno durante a busca."}), 500

# Endpoint de verificação de saúde para o Render
@app.route('/', methods=['GET'])
def health_check():
    return "Serviço de busca está no ar."

if __name__ == '__main__':
    initialize_knowledge_base()
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
