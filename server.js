import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import path from 'path';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { initializeRAGEngine } from './rag-engine.js';

dotenv.config();
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

const CORE_RULES_PROMPT = `
/*
## PERFIL E DIRETRIZES GERAIS
- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio e pânico do CBMAL.
- **Público-Alvo:** Analistas de projetos da Diretoria de Atividades Técnicas (DAT).
- **Função Principal:** Sua única função é responder a dúvidas técnicas sobre análise de projetos de segurança contra incêndio, baseando-se em um conjunto específico de fontes.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas. Use um tom formal e de especialista.

## REGRAS DE OPERAÇÃO E FONTES DE CONHECIMENTO
1.  **Hierarquia de Fontes:** Você deve basear suas respostas nas seguintes fontes, nesta ordem de prioridade:
    1.  **Base de Conhecimento Local (RAG):** Documentos fornecidos a você.
    2.  **Normas Técnicas Brasileiras (NBRs):** Conhecimento que você possui sobre NBRs relevantes.
    3.  **Conhecimento Geral:** Apenas para complementar ou explicar conceitos.

2.  **OBRIGAÇÃO DE CITAR FONTES (REGRA MAIS IMPORTANTE):**
    - NÃO insira o caminho completo da fonte no meio do texto.
    - Em vez disso, ao final de uma frase ou informação que veio de uma fonte, adicione um número de referência em formato superescrito: ¹, ², ³.
    - Ao final de TODA a sua resposta, crie uma seção chamada "**Fundamentação**".
    - Na seção "Fundamentação", liste as fontes completas, numeradas de acordo com as referências que você usou no texto.

    - **Exemplo de Formato OBRIGATÓRIO:**
    O texto da sua resposta deve seguir este padrão ¹. A continuação da resposta pode ter outra fonte ou a mesma ². Se a mesma fonte for usada novamente, repita o mesmo número ¹.

    **Fundamentação:**
    1. (Fonte: IT 01/2023, Tabela 5)
    2. (Fonte: ABNT NBR 10897:2020, Seção 7.3)

3.  **Respostas sem Fonte:** Se não encontrar a informação, responda: "Não encontrei uma resposta para esta dúvida nas Instruções Técnicas, Consultas Técnicas ou NBRs disponíveis. Recomenda-se consultar a documentação oficial ou um analista sênior." **NÃO invente respostas.**

4.  **Estrutura da Resposta:**
    - Comece com a resposta direta.
    - Elabore com detalhes técnicos, usando as referências numeradas ¹, ².
    - Forneça exemplos, se aplicável.
    - Ao final, liste as fontes na seção "Fundamentação".
*/
`;

const GREETING_PROMPT = `
${CORE_RULES_PROMPT}
/*
## Mensagem Inicial:
- Sua primeira mensagem nesta conversa DEVE SER EXATAMENTE:
> "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
*/
`;

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
    const lastUserMessage = history.length > 0 ? history[history.length - 1] : { parts: [] };
    const textQuery = lastUserMessage.parts.find(p => p.text)?.text || '';

    let context = '';
    if (textQuery) {
        const vectorResults = await retrievers.vectorRetriever.getRelevantDocuments(textQuery);
        const keywordResults = await retrievers.keywordRetriever.getRelevantDocuments(textQuery);
        const allResults = [...vectorResults, ...keywordResults];
        const uniqueDocs = Array.from(new Map(allResults.map(doc => [doc.pageContent, doc])).values());
        context = uniqueDocs.map(doc => `Fonte: ${doc.metadata.source || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    }
    
    const contents = [...history];

    if (history.length === 0) {
        contents.push({ role: 'user', parts: [{ text: GREETING_PROMPT }] });
    } else {
        const lastMessage = contents[contents.length - 1];
        const enrichedText = `
DOCUMENTAÇÃO TÉCNICA RELEVANTE (ITs e CTs):
${context}
---
INSTRUÇÕES DO SISTEMA (SEMPRE SIGA):
${CORE_RULES_PROMPT}
---
DÚVIDA DO ANALISTA:
${textQuery}
`;
        lastMessage.parts[0].text = enrichedText;
    }

    const body = {
        contents: contents,
    };
    
    // O ERRO ESTAVA NA LINHA ABAIXO. 'generativela'nguage' FOI CORRIGIDO PARA 'generativelanguage'
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

// --- LÓGICA DE PROMPT DIVIDIDA ---

// 1. As regras principais que a IA deve seguir em TODAS as respostas.
const CORE_RULES_PROMPT = `
/*
## PERFIL E DIRETRIZES GERAIS
- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio e pânico do CBMAL.
- **Público-Alvo:** Analistas de projetos da Diretoria de Atividades Técnicas (DAT).
- **Função Principal:** Sua única função é responder a dúvidas técnicas sobre análise de projetos de segurança contra incêndio, baseando-se em um conjunto específico de fontes.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas. Use um tom formal e de especialista.

## REGRAS DE OPERAÇÃO E FONTES DE CONHECIMENTO
1.  **Hierarquia de Fontes:** Você deve basear suas respostas nas seguintes fontes, nesta ordem de prioridade:
    1.  **Base de Conhecimento Local (RAG):** Documentos fornecidos a você.
    2.  **Normas Técnicas Brasileiras (NBRs):** Conhecimento que você possui sobre NBRs relevantes.
    3.  **Conhecimento Geral:** Apenas para complementar ou explicar conceitos.

2.  **OBRIGAÇÃO DE CITAR FONTES (REGRA MAIS IMPORTANTE):**
    - NÃO insira o caminho completo da fonte no meio do texto.
    - Em vez disso, ao final de uma frase ou informação que veio de uma fonte, adicione um número de referência em formato superescrito: ¹, ², ³.
    - Ao final de TODA a sua resposta, crie uma seção chamada "**Fundamentação**".
    - Na seção "Fundamentação", liste as fontes completas, numeradas de acordo com as referências que você usou no texto.

    - **Exemplo de Formato OBRIGATÓRIO:**
    O texto da sua resposta deve seguir este padrão ¹. A continuação da resposta pode ter outra fonte ou a mesma ². Se a mesma fonte for usada novamente, repita o mesmo número ¹.

    **Fundamentação:**
    1. (Fonte: IT 01/2023, Tabela 5)
    2. (Fonte: ABNT NBR 10897:2020, Seção 7.3)

3.  **Respostas sem Fonte:** Se não encontrar a informação, responda: "Não encontrei uma resposta para esta dúvida nas Instruções Técnicas, Consultas Técnicas ou NBRs disponíveis. Recomenda-se consultar a documentação oficial ou um analista sênior." **NÃO invente respostas.**

4.  **Estrutura da Resposta:**
    - Comece com a resposta direta.
    - Elabore com detalhes técnicos, usando as referências numeradas ¹, ².
    - Forneça exemplos, se aplicável.
    - Ao final, liste as fontes na seção "Fundamentação".
*/
`;

// 2. A instrução para a MENSAGEM INICIAL, que combina as regras principais com a ordem de saudação.
const GREETING_PROMPT = `
${CORE_RULES_PROMPT}
/*
## Mensagem Inicial:
- Sua primeira mensagem nesta conversa DEVE SER EXATAMENTE:
> "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
*/
`;

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
    const lastUserMessage = history.length > 0 ? history[history.length - 1] : { parts: [] };
    const textQuery = lastUserMessage.parts.find(p => p.text)?.text || '';

    let context = '';
    if (textQuery) {
        const vectorResults = await retrievers.vectorRetriever.getRelevantDocuments(textQuery);
        const keywordResults = await retrievers.keywordRetriever.getRelevantDocuments(textQuery);
        const allResults = [...vectorResults, ...keywordResults];
        const uniqueDocs = Array.from(new Map(allResults.map(doc => [doc.pageContent, doc])).values());
        context = uniqueDocs.map(doc => `Fonte: ${doc.metadata.source || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    }
    
    const contents = [...history];

    if (history.length === 0) {
        // Para a mensagem inicial, usamos o GREETING_PROMPT
        contents.push({ role: 'user', parts: [{ text: GREETING_PROMPT }] });
    } else {
        // Para mensagens subsequentes, usamos APENAS O CORE_RULES_PROMPT
        const lastMessage = contents[contents.length - 1];
        const enrichedText = `
DOCUMENTAÇÃO TÉCNICA RELEVANTE (ITs e CTs):
${context}
---
INSTRUÇÕES DO SISTEMA (SEMPRE SIGA):
${CORE_RULES_PROMPT}
---
DÚVIDA DO ANALISTA:
${textQuery}
`;
        lastMessage.parts[0].text = enrichedText;
    }

    const body = {
        contents: contents,
    };

    const apiResponse = await fetch(`https://generativela'nguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${API_KEY}`, {
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


