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
- **Função Principal:** Sua única função é responder a dúvidas técnicas sobre análise de projetos de segurança contra incêndio, baseando-se em um conjunto específico de fontes.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas.
---
## PROCESSO DE RACIOCÍNIO (SEMPRE SIGA ESTES PASSOS ANTES DE RESPONDER)
1.  **Decompor a Pergunta:** Analise a pergunta do usuário e extraia as palavras-chave e os critérios principais (ex: 'F-6', 'menos de 750m²', 'hospital', 'extintores').
2.  **Mapear com o Contexto (RAG):** Examine o contexto da base de conhecimento fornecido. Procure por títulos, tabelas ou seções que correspondam diretamente a essas palavras-chave.
3.  **Priorizar o Contexto Correto:** Se o contexto recuperado contiver informações conflitantes que dependem de um critério numérico (como área ou altura), você **DEVE OBRIGATORIAMENTE** usar a informação da seção que corresponde explicitamente à pergunta do usuário. Ignore os trechos que não se aplicam.
4.  **Dupla Checagem da Extração:** Antes de formular a resposta, revise sua própria extração de dados. **VERIFIQUE DUAS VEZES** se a informação pertence inequivocamente à coluna e linha corretas.
5.  **Sintetizar a Resposta:** Com base no trecho priorizado e verificado, construa sua resposta.
6.  **Citar Fontes:** Para cada informação, cite a fonte específica de onde ela foi retirada.
7.  **Fallback (Plano B):** **Apenas se**, após seguir rigorosamente os passos acima, a informação realmente não estiver presente, utilize a resposta padrão.
---
## REGRAS DE OPERAÇÃO
- **OBRIGAÇÃO DE CITAR FONTES:** TODA AFIRMAÇÃO TÉCNICA DEVE SER ACOMPANHADA DE SUA FONTE.
- **Mensagem Inicial:** Ao receber uma conversa vazia, sua ÚNICA resposta deve ser: "Bom dia, Analista. Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
*/
`;

let vectorStore;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/health', (req, res) => {
    res.status(200).send('Servidor do Assistente da DAT está ativo e saudável.');
});

app.post('/api/generate', async (req, res) => {
  const { history } = req.body;
  if (!history) {
    return res.status(400).json({ error: 'O histórico da conversa é obrigatório.' });
  }

  try {
    const isInitialMessage = history.length === 0;
    let context = "";

    if (!isInitialMessage) {
        const textQuery = history[history.length - 1]?.parts[0]?.text || '';
        if (textQuery && vectorStore) {
            const llm = new ChatGoogleGenerativeAI({ apiKey: API_KEY, modelName: API_MODEL, temperature: 0 });
            const retriever = MultiQueryRetriever.fromLLM({
                llm: llm,
                retriever: vectorStore.asRetriever(),
                verbose: true,
            });
            const contextDocs = await retriever.getRelevantDocuments(textQuery);
            context = contextDocs.map(doc => `Fonte: ${doc.metadata.source || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
        }

        // Adiciona o contexto do RAG à última mensagem do usuário
        const lastMessage = history[history.length - 1];
        lastMessage.parts.unshift({ text: `CONTEXTO DA BASE DE CONHECIMENTO:\n${context}\n---\nPERGUNTA DO ANALISTA:` });
    }

    const body = {
      contents: history, // Apenas o histórico da conversa
      systemInstruction: { // As instruções do sistema vão separadas
        parts: [{ text: SYSTEM_PROMPT }]
      }
    };

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
