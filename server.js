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
const API_MODEL = 'gemini-1.5-flash-latest';

if (!API_KEY) {
  console.error('[ERRO CRÍTICO] Variável de ambiente GEMINI_API_KEY não definida.');
  process.exit(1);
}

// Prompts Simplificados
const GREETING_ONLY_PROMPT = `Sua única tarefa é responder com a seguinte frase, e nada mais: "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."`;

const RESPONSE_GENERATION_PROMPT = `
/*
## PERFIL
- Você é o "Assistente Técnico da DAT". Sua precisão e fidelidade às fontes são absolutas.

## CONTEXTO FORNECIDO
- O contexto abaixo foi extraído de arquivos Markdown que contêm as Instruções Técnicas.
{CONTEXT}

## TAREFA
- Baseado EXCLUSIVAMENTE no CONTEXTO FORNECIDO, responda à DÚVIDA DO ANALISTA.
- Se o contexto contiver informações sobre classificação ou exigências, estruture a resposta de forma clara e objetiva.
- Se o contexto for sobre um conceito, explique-o de forma didática.
- Se o contexto estiver vazio ou não for relevante para a pergunta, informe que não encontrou informações precisas sobre o assunto na base de conhecimento.
- No final da resposta, SEMPRE adicione a seção "Fundamentação", citando a(s) fonte(s) do(s) trecho(s) que você usou, que estão no campo "source" do contexto.

## DÚVIDA DO ANALISTA
{QUERY}
*/
`;

let ragTools;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

async function callGemini(prompt) {
    const body = { contents: [{ role: 'user', parts: [{ text: prompt }] }] };
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${API_KEY}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60000)
    });
    if (!response.ok) {
        const errorText = await response.text();
        console.error("API Error Response:", errorText);
        throw new Error(`API Error: ${response.status}`);
    }
    const data = await response.json();
    if (!data.candidates || data.candidates.length === 0 || !data.candidates[0].content.parts) {
        console.error("Invalid API response structure:", data);
        throw new Error("A API retornou uma resposta inválida ou vazia.");
    }
    return data.candidates[0].content.parts[0].text;
}

app.post('/api/generate', async (req, res) => {
  const { history } = req.body;
  if (!history) {
    return res.status(400).json({ error: 'O histórico da conversa é obrigatório.' });
  }

  try {
    // 1. Lida com o início da conversa
    if (history.length === 0) {
        const reply = await callGemini(GREETING_ONLY_PROMPT);
        return res.json({ reply });
    }

    const textQuery = history[history.length - 1].parts[0].text;
    
    // 2. Busca diretamente os documentos relevantes na base de texto
    const contextDocs = await ragTools.textRetriever.getRelevantDocuments(textQuery);
    const context = contextDocs.map(doc => `Fonte: ${path.basename(doc.metadata.source)}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    
    // 3. Gera a resposta final com base no contexto encontrado
    const finalPrompt = RESPONSE_GENERATION_PROMPT
        .replace('{CONTEXT}', context || 'Contexto fornecido vazio.')
        .replace('{QUERY}', textQuery);
        
    const reply = await callGemini(finalPrompt);

    return res.json({ reply });

  } catch (error) {
    console.error(`[ERRO] Falha ao gerar resposta:`, error);
    res.status(503).json({ error: `Ocorreu um erro ao processar sua solicitação: ${error.message}` });
  }
});

async function startServer() {
  ragTools = await initializeRAGEngine();
  app.listen(PORT, () => {
    console.log(`Servidor do Assistente Técnico da DAT (versão RAG Puro) a rodar na porta ${PORT}.`);
  });
}

startServer();
