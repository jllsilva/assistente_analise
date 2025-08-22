import fs from 'fs/promises';
import path from 'path';
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

// Função auxiliar para encontrar a tabela de exigências correta
function getTabelaExigencias(knowledgeBase, grupo, area, altura) {
    // Lógica para Tabela 5
    if (area <= 750 && altura <= 12) {
        const tabela = knowledgeBase.tabela_5_exigencias_area_menor_igual_750;
        const regra = tabela.regras.find(r => r.criterio.includes(grupo.charAt(0))); // Busca pela letra do grupo
        if (regra) return { ...regra, tabela_info: tabela.tabela_info };
    } 
    // Lógica para Tabelas 6
    else {
        for (const key in knowledgeBase.tabelas_6_exigencias_area_maior_750) {
            const tabela6 = knowledgeBase.tabelas_6_exigencias_area_maior_750[key];
            if (tabela6.grupo_ocupacao.includes(grupo)) {
                const regra = tabela6.regras_por_altura.find(r => {
                    if ( (altura > 0 && altura <= 6) && (r.altura.includes("≤ 6") || r.classificacao === "Térrea") ) return true;
                    if ( (altura > 6 && altura <= 12) && r.altura.includes("6 < H ≤ 12") ) return true;
                    if ( (altura > 12 && altura <= 23) && r.altura.includes("12 < H ≤ 23") ) return true;
                    if ( (altura > 23 && altura <= 30) && r.altura.includes("23 < H ≤ 30") ) return true;
                    if ( altura > 30 && r.altura.includes("Acima de 30") ) return true;
                    return false;
                });
                if (regra) return { ...regra, tabela_info: tabela6.tabela_info };
            }
        }
    }
    return null;
}

export async function initializeRAGEngine() {
  try {
    // Carrega o JSON
    const jsonPath = path.resolve(process.cwd(), 'knowledge_base_json', 'knowledge_base.json');
    const jsonData = await fs.readFile(jsonPath, 'utf-8');
    const knowledgeBaseJSON = JSON.parse(jsonData);

    // Carrega os textos
    const textLoader = new DirectoryLoader('./knowledge_base_text', {
        '.doc': (path) => new DocxLoader(path),
        '.docx': (path) => new DocxLoader(path),
        '.md': (path) => new TextLoader(path),
    });
    const textDocs = await textLoader.load();
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 150 });
    const splits = await textSplitter.splitDocuments(textDocs);
    const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GEMINI_API_KEY });
    const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
    const textRetriever = vectorStore.asRetriever(5);
    
    console.log("[RAG Engine] Híbrido pronto.");

    // Retorna as ferramentas para o servidor
    return { knowledgeBaseJSON, textRetriever, getTabelaExigencias };

  } catch (error) {
    console.error('[RAG Engine] Erro ao inicializar motor RAG Híbrido:', error);
    throw error;
  }
}
