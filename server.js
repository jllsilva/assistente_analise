import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import path from 'path';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { initializeRAGEngine } from './rag-engine.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const API_KEY = process.env.GEMINI_API_KEY;
const API_MODEL = 'gemini-2.5-flash';

if (!API_KEY) {
  console.error('[ERRO CRÍTICO] Variável de ambiente GEMINI_API_KEY não definida.');
  process.exit(1);
}

const CORE_RULES_PROMPT = `
/*
## PERFIL E DIRETRIZES
- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio do CBMAL.
- **Fontes de Verdade:** Você possui dois tipos de fontes: DADOS ESTRUTURADOS (JSON) para fatos e classificações, e DOCUMENTOS DE TEXTO para contexto e definições.

## FLUXO DE RACIOCÍNIO HÍBRIDO
1.  **Análise da Pergunta:** Primeiro, entenda a intenção do usuário. Ele está pedindo uma classificação/exigência específica ou uma explicação/definição?
2.  **Estratégia de Busca:**
    - Para perguntas sobre **classificação ou exigências de tabelas** (ex: "qual o grupo de um hotel?", "exigências para F-6 com 100m²"), sua **PRIORIDADE MÁXIMA** é usar os DADOS ESTRUTURADOS (JSON) fornecidos. Eles são sua fonte de verdade para fatos.
    - Para perguntas **conceituais ou descritivas** (ex: "o que é compartimentação?", "explique a IT-17"), sua prioridade é usar os DOCUMENTOS DE TEXTO.
3.  **Formulação da Resposta:**
    - Use os dados da fonte mais apropriada para construir sua resposta. Se necessário, use ambos.
    - Se os dados JSON forem insuficientes (faltando área/altura), peça-os ao analista.
    - Mantenha o formato de citação (¹, ²) e a seção "Fundamentação", indicando a fonte correta.
*/
`;

const GREETING_ONLY_PROMPT = `Você é um assistente técnico do Corpo de Bombeiros de Alagoas. Sua única tarefa é responder com a seguinte frase, e nada mais: "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."`;

let retrievers;

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
    let contentsForApi;

    if (isInitialMessage) {
        contentsForApi = [{ role: 'user', parts: [{ text: GREETING_ONLY_PROMPT }] }];
    } else {
        const textQuery = history[history.length - 1].parts[0].text;
        
        // Realiza a busca em ambas as fontes
        const jsonDocs = await retrievers.jsonRetriever.getRelevantDocuments(textQuery);
        const textDocs = await retrievers.textRetriever.getRelevantDocuments(textQuery);

        const jsonContext = jsonDocs.map(doc => `Fonte Estruturada (JSON): ${doc.metadata.source}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
        const textContext = textDocs.map(doc => `Fonte Textual: ${doc.metadata.source}\nConteúdo: ${doc.pageContent}`).join('\n---\n');

        const enrichedText = `
DADOS ESTRUTURADOS (JSON) - Use como prioridade para classificações e exigências:
${jsonContext}
---
DOCUMENTOS DE TEXTO - Use para definições e contexto:
${textContext}
---
INSTRUÇÕES DO SISTEMA (SEMPRE SIGA):
${CORE_RULES_PROMPT}
---
DÚVIDA DO ANALISTA:
${textQuery}
`;
        const allButLast = history.slice(0, -1);
        contentsForApi = [
            ...allButLast,
            { role: 'user', parts: [{ text: enrichedText }] }
        ];
    }

    const body = {
        contents: contentsForApi,
    };
    
    const apiResponse = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60000)
    });

    if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.error?.message || `API Error: ${apiResponse.status}`);
    }

    const data = await apiResponse.json();

    if (!data.candidates || data.candidates.length === 0) {
        const feedback = data.promptFeedback;
        if (feedback && feedback.blockReason) {
            throw new Error(`Resposta bloqueada por segurança: ${feedback.blockReason}. ${feedback.blockReasonMessage || ''}`);
        }
        throw new Error("A API retornou uma resposta vazia.");
    }

    const reply = data.candidates[0].content.parts[0].text;

    if (reply === undefined || reply === null) {
      throw new Error("A API retornou uma resposta válida, mas sem texto.");
    }

    console.log(`[Sucesso] Resposta da API gerada para a DAT.`);
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
  retrievers = await initializeRAGEngine();
  app.listen(PORT, () => {
    console.log(`Servidor do Assistente Técnico da DAT a rodar na porta ${PORT}.`);
  });
}

startServer();
