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
- **Identidade:** Você é o "Assistente Técnico da DAT". Sua função é ser um porta-voz para os dados técnicos fornecidos.
- **Fonte da Verdade:** Sua ÚNICA fonte de informação são os dados JSON fornecidos no campo "pageContent". Esses dados são 100% corretos e pré-validados. É PROIBIDO usar conhecimento geral, fazer suposições ou alucinar informações.
- **Tarefa Principal:** Leia a pergunta do usuário e os dados JSON no "pageContent". Use os dados para formular uma resposta clara, profissional e baseada EXCLUSIVAMENTE nos fatos fornecidos.

## FLUXO DE TRABALHO
1.  **Análise dos Dados:** O "pageContent" contém uma "ficha" JSON. Analise seus campos.
2.  **Se for uma ficha de CLASSIFICAÇÃO:** Apresente a classificação encontrada (Grupo, Divisão, Descrição) e SEMPRE peça a "área construída" e a "altura da edificação" para poder determinar as exigências.
3.  **Se for uma ficha de EXIGÊNCIAS:** Apresente a lista de exigências de forma clara, usando bullets (*). Se uma exigência tiver uma "nota", mencione-a.
4.  **Citações:** Use o sistema de citação (¹, ²) e a seção "Fundamentação". A fonte é o campo "metadata.source".
*/
`;

const GREETING_ONLY_PROMPT = `Você é um assistente técnico do Corpo de Bombeiros de Alagoas. Sua única tarefa é responder com a seguinte frase, e nada mais: "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."`;

let retrievers;
let conversationState = {}; // Simples gerenciador de estado em memória

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
    let contextDocs = [];

    if (isInitialMessage) {
        conversationState = {}; // Limpa o estado para uma nova conversa
        contentsForApi = [{ role: 'user', parts: [{ text: GREETING_ONLY_PROMPT }] }];
    } else {
        const textQuery = history[history.length - 1].parts[0].text;
        
        // Lógica para extrair área e altura da resposta do usuário
        const areaMatch = textQuery.match(/(\d+)\s*m/);
        const alturaMatch = textQuery.match(/(\d+)\s*m/);
        const area = areaMatch ? parseInt(areaMatch[1], 10) : null;
        const altura = alturaMatch ? parseInt(alturaMatch[1], 10) : null;

        if (conversationState.grupoPendente && area && altura) {
            // Se estávamos esperando área/altura, busca pelas exigências
            contextDocs = await retrievers.jsonRetriever.getRelevantDocuments(
                "", { grupo: conversationState.grupoPendente, area, altura }
            );
            conversationState = {}; // Limpa o estado
        } else {
            // Senão, busca por uma nova classificação
            contextDocs = await retrievers.jsonRetriever.getRelevantDocuments(textQuery);
            if (contextDocs.length > 0 && contextDocs[0].metadata.tipo === 'classificacao') {
                const classData = JSON.parse(contextDocs[0].pageContent);
                conversationState = { grupoPendente: classData.grupo }; // Salva o grupo pendente
            }
        }
        
        const context = contextDocs.map(doc => `Fonte: ${doc.metadata.source}\nConteúdo JSON: ${doc.pageContent}`).join('\n---\n');

        const enrichedText = `
DADOS TÉCNICOS ESTRUTURADOS (100% CORRETOS):
${context}
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
