import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
// Importa o novo divisor de texto especialista em Markdown
import { MarkdownHeaderTextSplitter } from "langchain/text_splitter";
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

    // --- NOVA ESTRATÉGIA DE DIVISÃO DE TEXTO ---
    // Define a estrutura de títulos que o divisor deve respeitar
    const headersToSplitOn = [
        ["#", "Header1"],
        ["##", "Header2"],
        ["###", "Header3"],
    ];

    // Cria uma instância do divisor de Markdown
    const markdownSplitter = new MarkdownHeaderTextSplitter({
        headersToSplitOn: headersToSplitOn,
    });

    // Divide os documentos usando a nova estratégia
    const splits = await markdownSplitter.splitDocuments(docs);
    console.log(`[RAG Engine] Documentos divididos em ${splits.length} chunks usando a estrutura do Markdown.`);
    // --- FIM DA NOVA ESTRATÉGIA ---


    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        modelName: "text-embedding-004"
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);

    console.log(`[RAG Engine] Indexação concluída. ${splits.length} pedaços de texto carregados na memória.`);

    return vectorStore.asRetriever({ k: 5 }); // Aumentei para 5 chunks para dar mais contexto

  } catch (error) {
    console.error('[RAG Engine] Falha ao inicializar a base de conhecimento:', error);
    return { getRelevantDocuments: () => Promise.resolve([]) };
  }
}
