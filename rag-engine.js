import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export async function initializeRAGEngine() {
  try {
    console.log('[RAG Engine] Iniciando indexação da base de conhecimento...');
    
    const loader = new DirectoryLoader(
      './knowledge_base',
      {
        // ADICIONAMOS O SUPORTE PARA ARQUIVOS .doc AQUI
        '.doc': (path) => new DocxLoader(path),
        '.docx': (path) => new DocxLoader(path),
        '.md': (path) => new TextLoader(path),
        '.pdf': (path) => new PDFLoader(path, { splitPages: true }),
      }
    );
    const docs = await loader.load();

    if (docs.length === 0) {
      console.log('[RAG Engine] Nenhum documento encontrado.');
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
    const vectorRetriever = vectorStore.asRetriever({ k: 6 });

    console.log('[RAG Engine] Criando buscador de keyword MANUAL...');
    
    const keywordRetriever = {
      getRelevantDocuments: async (query) => {
        const queryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);
        if (queryTerms.length === 0) {
          return [];
        }
        const relevantDocs = splits.filter(doc => {
          const content = doc.pageContent.toLowerCase();
          return queryTerms.some(term => content.includes(term));
        });
        return relevantDocs.slice(0, 6);
      }
    };

    console.log(`[RAG Engine] Indexação e preparação dos buscadores concluída.`);
    return { vectorRetriever, keywordRetriever };

  } catch (error) {
    console.error('[RAG Engine] Falha ao inicializar a base de conhecimento:', error);
    const emptyRetriever = { getRelevantDocuments: async () => [] };
    return { vectorRetriever: emptyRetriever, keywordRetriever: emptyRetriever };
  }
}
