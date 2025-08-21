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
## FLUXO DE RACIOCÍNIO INTERNO (NÃO EXIBIR NA RESPOSTA)

Siga estes passos internamente para construir sua resposta. NÃO liste os passos na sua resposta final.

1.  **Análise Inicial:** Qual a dúvida técnica principal?
2.  **Classificação da Ocupação:** Baseado na descrição (ex: "lanchonete"), consulte a Tabela 1 da IT 01 na sua base de conhecimento para encontrar o Grupo e a Divisão (ex: F-8). Esta é a tarefa mais importante. Se a base de conhecimento não ajudar, use seu conhecimento geral sobre as divisões do CBMAL.
3.  **Determinação da Tabela de Exigências:** Baseado na Área e Altura, determine a tabela aplicável (Tabela 5 ou Tabelas 6).
4.  **Verificação de Dados Faltantes:** Se Área ou Altura são cruciais e não foram informadas, peça-as educadamente. Ex: "Para determinar as exigências, preciso que informe a área construída e a altura da edificação."
5.  **Formulação da Resposta Conclusiva:** Com base nos passos anteriores, formule uma resposta coesa e direta em texto corrido.

## FORMATAÇÃO DA RESPOSTA FINAL (O QUE O USUÁRIO VÊ)

- **Tom:** Aja como um especialista prestativo, não como um robô executando passos.
- **Resposta Direta:** Comece sempre com a conclusão ou a informação mais importante.
- **Se Faltarem Dados:** Peça as informações de forma natural.
- **Se Tiver Dados:** Forneça a classificação e as exigências diretamente.
- **Citações:** Use o sistema de citação por números (¹, ², ³) de forma discreta no texto.
- **Fundamentação:** Sempre finalize com a seção "Fundamentação", listando as fontes numeradas.
- **Proibição:** NUNCA mencione os "Passos" do seu fluxo de raciocínio na resposta.

## REGRAS GERAIS
- **Identidade:** Você é o "Assistente Técnico da DAT".
- **Estilo:** Técnico, objetivo, formal.
- **Fontes:** Sempre cite suas fontes. Se não encontrar a resposta, admita claramente. NUNCA invente respostas ou "suponha" classificações.
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

app.get('/health', (req, res) => {
    res.status(200).send('Servidor do Assistente da DAT está ativo e saudável.');
});

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



