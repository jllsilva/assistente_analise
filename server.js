import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
// Voltando a usar o splitter padrão e estável
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Esta função irá inicializar todo o nosso motor de busca
export async function initializeRAGEngine() {
  try {
    console.log('[RAG Engine] Iniciando indexação da base de conhecimento...');
    
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
      console.log('[RAG Engine] Nenhum documento encontrado. O servidor continuará sem conhecimento de RAG.');
      return { getRelevantDocuments: () => Promise.resolve([]) };
    }

    // Usando o RecursiveCharacterTextSplitter que é compatível com a sua versão do LangChain
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1500, // Tamanho de cada "pedaço" de texto
      chunkOverlap: 200, // Sobreposição entre os pedaços para não perder o contexto
    });

    const splits = await textSplitter.splitDocuments(docs);
    console.log(`[RAG Engine] Documentos divididos em ${splits.length} chunks.`);
    
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        modelName: "text-embedding-004"
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

    console.log(`[RAG Engine] Indexação concluída. ${splits.length} pedaços de texto carregados na memória.`);

    return vectorStore.asRetriever({ k: 5 });

  } catch (error) {
    console.error('[RAG Engine] Falha ao inicializar a base de conhecimento:', error);
    return { getRelevantDocuments: () => Promise.resolve([]) };
  }
}
