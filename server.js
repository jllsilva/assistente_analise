import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import path from 'path';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { createVectorStore } from './rag-engine.js';

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
2.  **Mapear com o Contexto (RAG):** Examine o contexto da base de conhecimento fornecido. A fonte mais relevante provavelmente estará em um arquivo cujo nome corresponde às palavras-chave.
3.  **Priorizar o Contexto Correto:** Se o contexto recuperado contiver informações conflitantes que dependem de um critério numérico (como área ou altura), você **DEVE OBRIGATORIAMENTE** usar a informação da seção que corresponde explicitamente à pergunta do usuário.
4.  **Dupla Checagem da Extração:** Antes de formular a resposta, revise sua própria extração de dados. **VERIFIQUE DUAS VEZES** se a informação pertence inequivocamente à coluna e linha corretas.
5.  **Sintetizar a Resposta:** Com base no trecho priorizado e verificado, construa sua resposta.
6.  **Citar Fontes:** Para cada informação, cite a fonte. O nome do arquivo (ex: IT_01_Tabela_5_Area_Menor_750.md) é uma excelente fonte.
7.  **Fallback (Plano B):** **Apenas se**, após seguir rigorosamente os passos acima, a informação realmente não estiver presente, utilize a resposta padrão.
---
## REGRAS DE OPERAÇÃO
- **Mensagem Inicial:** Ao receber uma conversa vazia, sua ÚNICA resposta deve ser: "Bom dia, Analista. Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
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
        const retriever = vectorStore.asRetriever();
        const contextDocs = await retriever.getRelevantDocuments(textQuery);
        context = contextDocs.map(doc => `Nome do Arquivo Fonte: ${doc.metadata.source?.split('/').pop() || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    }

    const body = {
      contents: history,
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] }
    };

    if (!isInitialMessage) {
        body.contents[body.contents.length - 1].parts.unshift({ text: `\nCONTEXTO DA BASE DE CONHECIMENTO:\n${context}\n---\n` });
    }

    const apiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(30000)
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
