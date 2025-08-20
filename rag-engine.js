import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Esta função irá inicializar todo o nosso motor de busca
export async function initializeRAGEngine() {
  try {import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export async function createVectorStore() {
  try {
    console.log('[Vector Store] Iniciando indexação da base de conhecimento...');
    
    const loader = new DirectoryLoader(
      './knowledge_base',
      {
        '.pdf': (path) => new PDFLoader(path, { splitPages: false }),
        '.docx': (path) => new DocxLoader(path),
        '.md': (path) => new TextLoader(path),
      }
    );
    const docs = await loader.load();

    if (docs.length === 0) {
      console.log('[Vector Store] Nenhum documento encontrado.');
      return await MemoryVectorStore.fromTexts(["base de conhecimento vazia"], {}, new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GEMINI_API_KEY }));
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1500,
      chunkOverlap: 200,
    });

    const splits = await textSplitter.splitDocuments(docs);
    console.log(`[Vector Store] Documentos divididos em ${splits.length} chunks.`);
    
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        modelName: "text-embedding-004"
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
    console.log(`[Vector Store] Indexação concluída.`);
    return vectorStore;

  } catch (error) {
    console.error('[Vector Store] Falha ao inicializar a base de conhecimento:', error);
    return await MemoryVectorStore.fromTexts(["falha ao carregar base de conhecimento"], {}, new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GEMINI_API_KEY }));
  }
}
    console.log('[RAG Engine] Iniciando indexação da base de conhecimento...');
    
    const loader = new DirectoryLoader(
      './knowledge_base',
      {
        '.pdf': (path) => new PDFLoader(path, { splitPages: false }),
        '.docx': (path) => new DocxLoader(path),
      }
    );
    const docs = await loader.load();

    if (docs.length === 0) {
      console.log('[RAG Engine] Nenhum documento encontrado na base de conhecimento. O servidor continuará sem conhecimento de RAG.');
      return { getRelevantDocuments: () => Promise.resolve([]) };
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1500,
      chunkOverlap: 200,
    });
    const splits = await textSplitter.splitDocuments(docs);

    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        modelName: "text-embedding-004"
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

    console.log(`[RAG Engine] Indexação concluída. ${splits.length} pedaços de texto carregados na memória.`);

    return vectorStore.asRetriever({ k: 4 });

  } catch (error) {
    console.error('[RAG Engine] Falha ao inicializar a base de conhecimento:', error);
    return { getRelevantDocuments: () => Promise.resolve([]) };
  }
}
