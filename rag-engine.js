import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// IMPORTAÇÕES CORRIGIDAS
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { EnsembleRetriever } from "@langchain/community/retrievers/ensemble"; // <-- ESTA LINHA FOI CORRIGIDA

// Esta função irá inicializar todo o nosso motor de busca
export async function initializeRAGEngine() {
  try {
    console.log('[RAG Engine] Iniciando indexação da base de conhecimento...');
    
    const loader = new DirectoryLoader(
      './knowledge_base',
      {
        '.pdf': (path) => new PDFLoader(path, { splitPages: true }),
        '.docx': (path) => new DocxLoader(path),
      }
    );
    const docs = await loader.load();

    if (docs.length === 0) {
      console.log('[RAG Engine] Nenhum documento encontrado na base de conhecimento. O servidor continuará sem conhecimento de RAG.');
      return { getRelevantDocuments: () => Promise.resolve([]) };
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 900,
      chunkOverlap: 150,
    });
    const splits = await textSplitter.splitDocuments(docs);

    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        modelName: "text-embedding-004"
    });

    console.log('[RAG Engine] Criando buscador vetorial em memória...');
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

    console.log('[RAG Engine] Criando buscador de keyword (BM25) em memória...');
    
    // 1. O Buscador de Keyword (BM25)
    const bm25Retriever = new BM25Retriever({
        vectorStore: undefined,
        documents: splits,
        k: 6,
    });

    // 2. O Buscador Vetorial (Semântico)
    const vectorStoreRetriever = vectorStore.asRetriever({ k: 6 });

    console.log('[RAG Engine] Combinando buscadores com o Ensemble Retriever...');

    // 3. O "Maestro" (Ensemble Retriever)
    const ensembleRetriever = new EnsembleRetriever({
        retrievers: [bm25Retriever, vectorStoreRetriever],
        weights: [0.5, 0.5],
    });

    console.log(`[RAG Engine] Indexação HÍBRIDA concluída. ${splits.length} pedaços de texto carregados na memória.`);

    return ensembleRetriever;

  } catch (error) {
    console.error('[RAG Engine] Falha ao inicializar a base de conhecimento:', error);
    return { getRelevantDocuments: () => Promise.resolve([]) };
  }
}
