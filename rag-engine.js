import fs from 'fs/promises';
import path from 'path';
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { BaseRetriever } from "@langchain/core/retrievers";

// Classe customizada para buscar nos dados JSON, seguindo o padrão LangChain
class JSONRetriever extends BaseRetriever {
  constructor(data) {
    super();
    this.data = data;
  }
  
  async _getRelevantDocuments(query) {
    const lowerQuery = query.toLowerCase();
    const relevantFichas = this.data.filter(ficha => {
      const searchableText = (ficha.busca || JSON.stringify(ficha)).toLowerCase();
      return searchableText.includes(lowerQuery);
    });
    
    // Converte as fichas encontradas para o formato de Documento que o sistema espera
    return relevantFichas.map(ficha => ({
      pageContent: JSON.stringify(ficha),
      metadata: { source: ficha.fonte, tipo: 'json' }
    }));
  }
}

export async function initializeRAGEngine() {
  try {
    // --- Braço 1: DADOS ESTRUTURADOS (JSON) ---
    console.log('[RAG Engine] Carregando e preparando o buscador JSON...');
    const jsonPath = path.resolve(process.cwd(), 'knowledge_base_json', 'knowledge_base_final.json');
    const jsonData = await fs.readFile(jsonPath, 'utf-8');
    const knowledgeBaseJSON = JSON.parse(jsonData);
    const jsonRetriever = new JSONRetriever(knowledgeBaseJSON);
    console.log(`[RAG Engine] Buscador JSON pronto com ${knowledgeBaseJSON.length} fichas.`);

    // --- Braço 2: DOCUMENTOS DE TEXTO (DOCX, MD, PDF) ---
    console.log('[RAG Engine] Carregando e preparando o buscador de Texto...');
    const textLoader = new DirectoryLoader('./knowledge_base_text', {
      '.doc': (path) => new DocxLoader(path),
      '.docx': (path) => new DocxLoader(path),
      '.md': (path) => new TextLoader(path),
      '.pdf': (path) => new PDFLoader(path),
    });
    
    const textDocs = await textLoader.load();
    if (textDocs.length > 0) {
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
        const splits = await textSplitter.splitDocuments(textDocs);
        const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GEMINI_API_KEY });
        const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
        const textRetriever = vectorStore.asRetriever(5); // Pega os 5 melhores resultados semânticos
        console.log(`[RAG Engine] Buscador de Texto pronto com ${splits.length} trechos.`);
        return { jsonRetriever, textRetriever };
    } else {
        console.log('[RAG Engine] Nenhum documento de texto encontrado. O buscador de texto ficará inativo.');
        const emptyRetriever = { getRelevantDocuments: async () => [] };
        return { jsonRetriever, textRetriever: emptyRetriever };
    }

  } catch (error) {
    console.error('[RAG Engine] Erro fatal ao inicializar o motor RAG Híbrido:', error);
    const emptyRetriever = { getRelevantDocuments: async () => [] };
    return { jsonRetriever: emptyRetriever, textRetriever: emptyRetriever };
  }
}
