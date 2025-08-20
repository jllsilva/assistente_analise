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
// PROMPT DO SISTEMA: Assistente Técnico da DAT - CBMAL (Versão 2.2 com Dupla Checagem)
/*
## PERFIL E DIRETRIZES GERAIS
- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio e pânico do CBMAL.
- **Função Principal:** Sua única função é responder a dúvidas técnicas sobre análise de projetos de segurança contra incêndio, baseando-se no contexto fornecido.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas.
---
## PROCESSO DE RACIOCÍNIO (SEMPRE SIGA ESTES PASSOS ANTES DE RESPONDER)
1.  **Analisar o Contexto:** Examine o contexto da base de conhecimento fornecido. O nome do arquivo fonte já é uma pista importante.
2.  **Priorizar o Contexto Correto:** Se o contexto contiver informações de múltiplas fontes, você **DEVE OBRIGATORIAMENTE** usar a informação da fonte que corresponde mais explicitamente à pergunta do usuário (ex: um arquivo sobre "área menor que 750m²" é mais importante que um sobre "área maior que 750m²").
3.  **Dupla Checagem da Extração:** Antes de formular a resposta, **VERIFIQUE DUAS VEZES** se a informação extraída pertence inequivocamente aos critérios da pergunta (ex: Grupo 'F-6').
4.  **Sintetizar a Resposta:** Com base no trecho priorizado e verificado, construa sua resposta.
5.  **Citar Fontes:** Para cada informação, cite a fonte. O nome do arquivo (ex: IT_01_Tabela_5.md) é uma excelente fonte.
6.  **Fallback (Plano B):** Se, após seguir rigorosamente os passos, a informação realmente não estiver presente, utilize a resposta padrão.
---
## REGRAS DE OPERAÇÃO
- **Mensagem Inicial:** Ao receber uma conversa vazia, sua ÚNICA resposta deve ser: "Saudações, sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
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

  try {
    const isInitialMessage = history.length === 0;
    let context = "";

    if (!isInitialMessage && vectorStore) {
        const textQuery = history[history.length - 1]?.parts[0]?.text || '';
        
        // --- LÓGICA DE BUSCA INTELIGENTE ---
        const llm = new ChatGoogleGenerativeAI({ apiKey: API_KEY, modelName: API_MODEL, temperature: 0 });
        const retriever = MultiQueryRetriever.fromLLM({
            llm: llm,
            retriever: vectorStore.asRetriever(5), // Busca 5 documentos iniciais
            verbose: true, // Deixe true para ver as queries geradas no log do Render
        });

        const contextDocs = await retriever.getRelevantDocuments(textQuery);
        context = contextDocs.map(doc => `Nome do Arquivo Fonte: ${doc.metadata.source?.split(/[\\/]/).pop() || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    }
    
    const contents = JSON.parse(JSON.stringify(history));

    const body = {
      contents: contents,
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] }
    };

    if (!isInitialMessage) {
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
  if (vectorStore) {
    app.listen(PORT, () => {
        console.log(`Servidor do Assistente Técnico da DAT a rodar na porta ${PORT}.`);
    });
  } else {
    console.log("A base de vetores está vazia, mas o servidor irá iniciar.");
    app.listen(PORT, () => {
        console.log(`Servidor do Assistente Técnico da DAT a rodar na porta ${PORT}, mas sem base de conhecimento.`);
    });
  }
}

startServer();

