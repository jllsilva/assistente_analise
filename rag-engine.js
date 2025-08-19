import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
// CORREÇÃO FINAL E DEFINITIVA: Importando do pacote correto que instalamos.
import { MarkdownHeaderTextSplitter } from "@langchain/textsplitters";
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

    const headersToSplitOn = [
        ["#", "Header1"],
        ["##", "Header2"],
        ["###", "Header3"],
    ];

    const markdownSplitter = new MarkdownHeaderTextSplitter({
        headersToSplitOn: headersToSplitOn,
    });

    const splits = await markdownSplitter.splitDocuments(docs);
    console.log(`[RAG Engine] Documentos divididos em ${splits.length} chunks usando a estrutura do Markdown.`);
    
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
