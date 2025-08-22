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
    
    // Lógica de busca na Tabela 1
    const classificacoes = this.data.tabela_1_classificacao_ocupacao.classificacoes;
    for (const grupo of classificacoes) {
        for (const divisao of grupo.divisoes) {
            if (divisao.busca && divisao.busca.includes(lowerQuery)) {
                return [{
                    pageContent: JSON.stringify({ ...divisao, grupo: grupo.grupo, ocupacao_uso: grupo.ocupacao_uso }),
                    metadata: { source: this.data.tabela_1_classificacao_ocupacao.tabela_info, tipo: 'classificacao' }
                }];
            }
        }
    }
    return [];
  }
}

export async function initializeRAGEngine() {
  try {
    // --- Braço 1: DADOS ESTRUTURADOS (JSON) ---
    console.log('[RAG Engine] Carregando buscador JSON...');
    const jsonPath = path.resolve(process.cwd(), 'knowledge_base_json', 'knowledge_base.json');
    const jsonData = await fs.readFile(jsonPath, 'utf-8');
    const knowledgeBaseJSON = JSON.parse(jsonData);
    const jsonRetriever = new JSONRetriever(knowledgeBaseJSON);
    console.log(`[RAG Engine] Buscador JSON pronto.`);

    // --- Braço 2: DOCUMENTOS DE TEXTO (DOCX, MD, PDF) ---
    console.log('[RAG Engine] Carregando buscador de Texto...');
    const textLoader = new DirectoryLoader('./knowledge_base_text', {
      '.doc': (path) => new DocxLoader(path),
      '.docx': (path) => new DocxLoader(path),
      '.md': (path) => new TextLoader(path),
      '.pdf': (path) => new PDFLoader(path),
    });
    
    const textDocs = await textLoader.load();
    let textRetriever;

    if (textDocs.length > 0) {
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
        const splits = await textSplitter.splitDocuments(textDocs);
        const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GEMINI_API_KEY });
        const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
        textRetriever = vectorStore.asRetriever(5);
        console.log(`[RAG Engine] Buscador de Texto pronto com ${splits.length} trechos.`);
    } else {
        console.log('[RAG Engine] Nenhum documento de texto encontrado. Buscador de texto inativo.');
        textRetriever = { getRelevantDocuments: async () => [] };
    }

    return { jsonRetriever, textRetriever };

  } catch (error) {
    console.error('[RAG Engine] Erro ao inicializar motor RAG Híbrido:', error);
    const emptyRetriever = { getRelevantDocuments: async () => [] };
    return { jsonRetriever: emptyRetriever, textRetriever: emptyRetriever };
  }
}
