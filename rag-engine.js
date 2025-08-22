import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

export async function initializeRAGEngine() {
  try {
    console.log("[RAG Engine] Inicializando motor de busca vetorial...");

    // Carrega todos os documentos da pasta de conhecimento textual.
    const textLoader = new DirectoryLoader('./knowledge_base_text', {
        '.md': (path) => new TextLoader(path),
        '.docx': (path) => new DocxLoader(path),
    });
    
    const textDocs = await textLoader.load();
    console.log(`[RAG Engine] ${textDocs.length} documentos carregados.`);

    // Divide os documentos em pedaços menores (chunks).
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1500, chunkOverlap: 200 });
    const splits = await textSplitter.splitDocuments(textDocs);
    console.log(`[RAG Engine] Documentos divididos em ${splits.length} chunks.`);

    // Cria os embeddings e o vector store em memória.
    const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GEMINI_API_KEY });
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
    
    // Cria o retriever que buscará os 5 chunks mais relevantes.
    const textRetriever = vectorStore.asRetriever(5);
    
    console.log("[RAG Engine] Motor de busca vetorial pronto.");

    // Retorna apenas o retriever para o servidor.
    return { textRetriever };

  } catch (error) {
    console.error('[RAG Engine] Erro ao inicializar motor RAG:', error);
    throw error;
  }
}
