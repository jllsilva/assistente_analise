import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import path from 'path';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { createVectorStore } from './rag-engine.js';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GEMINI_API_KEY;
const API_MODEL = 'gemini-2.5-flash-preview-05-20';

if (!API_KEY) {
  console.error('[ERRO CRÍTICO] Variável de ambiente GEMINI_API_KEY não definida.');
  process.exit(1);
}

const SYSTEM_PROMPT = `
// PROMPT DO SISTEMA: Assistente Técnico da DAT - CBMAL (Versão 3.0 com Busca Guiada)
/*
## PERFIL E DIRETRIZES GERAIS
- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio e pânico do CBMAL.
- **Função Principal:** Responder a dúvidas técnicas sobre análise de projetos, baseando-se no contexto da base de conhecimento fornecida.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas.
---
## ESTRATÉGIA DE RACIOCÍNIO E BUSCA GUIADA (SEMPRE SIGA ESTES PASSOS)
1.  **DECOMPOR A PERGUNTA (PENSAR):** Analise a pergunta do analista e extraia as "coordenadas" essenciais: Ocupação/Grupo, Área, Altura, e a Medida de Segurança específica.
2.  **HIERARQUIA DE BUSCA (PESQUISAR):** Com as coordenadas, busque no contexto priorizando arquivos da IT 01 e suas tabelas, usando os nomes dos arquivos .md como pista.
3.  **PRIORIZAR E FILTRAR (ANALISAR):** Se a busca retornar múltiplos documentos conflitantes, a fonte que corresponder mais precisamente às coordenadas de Área e Altura da pergunta é a correta. Ignore as outras.
4.  **SINTETIZAR A RESPOSTA (RESPONDER):** Com base nas informações corretas, construa uma resposta completa, detalhando cada exigência e citando a fonte.
5.  **PLANO B (FALLBACK):** Apenas se não encontrar uma correspondência, utilize a resposta padrão de "Não encontrei a informação...".
---
## REGRAS ADICIONAIS
- **Mensagem Inicial:** Ao receber uma conversa vazia, sua ÚNICA resposta deve ser:
    > "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
*/
`;

let vectorStore;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/generate', async (req, res) => {
  const { history } = req.body;
  if (!history) {
    return res.status(400).json({ error: 'O histórico da conversa é obrigatório.' });
  }

  const isInitialMessage = history.length === 0;
  if (isInitialMessage) {
    const initialMessage = "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos.";
    return res.json({ reply: initialMessage });
  }

  try {
    let context = "";
    if (vectorStore) {
        const textQuery = history[history.length - 1]?.parts[0]?.text || '';
        
        // --- INÍCIO DA LÓGICA DE BUSCA INTELIGENTE ---
        const llm = new ChatGoogleGenerativeAI({ apiKey: API_KEY, modelName: API_MODEL, temperature: 0 });
        const retriever = MultiQueryRetriever.fromLLM({
            llm: llm,
            retriever: vectorStore.asRetriever(5),
            verbose: true, 
        });

        const contextDocs = await retriever.getRelevantDocuments(textQuery);
        context = contextDocs.map(doc => `Nome do Arquivo Fonte: ${doc.metadata.source?.split(/[\\/]/).pop() || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
        // --- FIM DA LÓGICA DE BUSCA INTELIGENTE ---
    }
    
    const contents = JSON.parse(JSON.stringify(history));
    const body = {
      contents: contents,
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] }
    };

    if (body.contents.length > 0) {
        body.contents[body.contents.length - 1].parts.unshift({ text: `\nCONTEXTO DA BASE DE CONHECIMENTO:\n${context}\n---\n` });
    }

    const apiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(45000)
    });

    if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.error?.message || `API Error: ${apiResponse.status}`);
    }

    const data = await apiResponse.json();
    const reply = data.candidates?.[0]?.content?.parts?.[0]?.text;

    if (!reply) {
      throw new Error("A API retornou uma resposta válida, mas sem texto.");
    }

    return res.json({ reply });

  } catch (error) {
    console.error(`[ERRO] Falha ao gerar resposta:`, error);
    res.status(503).json({ error: `Ocorreu um erro ao processar sua solicitação: ${error.message}` });
  }
});

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

async function startServer() {
  vectorStore = await createVectorStore();
  app.listen(PORT, () => {
    console.log(`Servidor do Assistente Técnico da DAT a rodar na porta ${PORT}.`);
  });
}

startServer();
