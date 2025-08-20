import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";

// NOTE: REMOVEMOS A IMPORTAÇÃO DO 'EnsembleRetriever'

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
      console.log('[RAG Engine] Nenhum documento encontrado.');
      // Retornamos retrievers vazios para evitar erros no servidor
      const emptyRetriever = { getRelevantDocuments: async () => [] };
      return { vectorRetriever: emptyRetriever, keywordRetriever: emptyRetriever };
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

    console.log('[RAG Engine] Criando buscadores em memória...');
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
    
    const keywordRetriever = new BM25Retriever({
        vectorStore: undefined,
        documents: splits,
        k: 6,
    });

    const vectorRetriever = vectorStore.asRetriever({ k: 6 });

    console.log(`[RAG Engine] Indexação e preparação dos buscadores concluída.`);

    // Agora, em vez de um "maestro", retornamos os dois buscadores separados.
    return { vectorRetriever, keywordRetriever };

  } catch (error) {
    console.error('[RAG Engine] Falha ao inicializar a base de conhecimento:', error);
    const emptyRetriever = { getRelevantDocuments: async () => [] };
    return { vectorRetriever: emptyRetriever, keywordRetriever: emptyRetriever };
  }
}
