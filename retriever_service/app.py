import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env na pasta raiz
load_dotenv('../.env')

# Configura a chave de API para as bibliotecas do Google
# É crucial que a GEMINI_API_KEY esteja no seu arquivo .env ou nas variáveis de ambiente do Render
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Variável de ambiente GEMINI_API_KEY não encontrada.")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# --- Carrega a base de conhecimento na inicialização ---
vector_store = None

def initialize_knowledge_base():
    global vector_store
    try:
        print("[Retriever Service] Iniciando indexação da base de conhecimento...")
        # O caminho aponta para a pasta um nível acima
        knowledge_base_path = '../knowledge_base/'
        
        loader = DirectoryLoader(
            knowledge_base_path,
            glob="**/[!.]*",  # Padrão para pegar todos os arquivos (pdf, md, etc)
            use_multithreading=True,
            # Define o carregador a ser usado com base na extensão do arquivo
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
    global vector_store
    if not vector_store:
        return jsonify({"error": "A base de conhecimento não está disponível ou não foi carregada."}), 503

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "A 'query' (pergunta) é obrigatória no corpo da requisição."}), 400

    query = data['query']
    print(f"[Retriever Service] Recebida a query: {query}")

    try:
        # Usando a busca de similaridade simples e eficaz do vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Busca os 5 trechos mais relevantes
        docs = retriever.invoke(query)
        
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
    # O Render vai fornecer a porta através da variável de ambiente PORT
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
