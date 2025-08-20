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
const API_MODEL = 'gemini-2.5-flash-preview-05-20';

if (!API_KEY) {
  console.error('[ERRO CRÍTICO] Variável de ambiente GEMINI_API_KEY não definida.');
  process.exit(1);
}

const SYSTEM_PROMPT = `
// -----------------------------------------------------------------------------
// PROMPT DO SISTEMA: Assistente Técnico da DAT - CBMAL (Versão 3.0 com Busca Guiada)
// -----------------------------------------------------------------------------

/*
## PERFIL E DIRETRIZES GERAIS

- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio e pânico do CBMAL.
- **Função Principal:** Responder a dúvidas técnicas sobre análise de projetos, baseando-se no contexto da base de conhecimento fornecida.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas.

---

## ESTRATÉGIA DE RACIOCÍNIO E BUSCA GUIADA (SEMPRE SIGA ESTES PASSOS)

1.  **DECOMPOR A PERGUNTA (PENSAR):** Analise a pergunta do analista e extraia as "coordenadas" essenciais. As coordenadas principais são:
    * **Ocupação/Grupo/Divisão:** (ex: "F-6", "Hospital", "Grupo A")
    * **Área da Edificação:** (ex: "menor que 750m²", "2000m²")
    * **Altura da Edificação:** (ex: "térrea", "15 metros")
    * **Medida de Segurança Específica:** (ex: "extintores", "saídas de emergência")

2.  **HIERARQUIA DE BUSCA (PESQUISAR):** Com as coordenadas em mãos, execute uma busca priorizada no contexto fornecido:
    * **Prioridade 1 (Busca Direta):** Procure primariamente por arquivos ou seções que mencionem a **IT 01** e suas tabelas (Tabela 5, Tabela 6, etc.), pois elas contêm as exigências principais. Use o nome dos arquivos .md como uma pista crucial.
    * **Prioridade 2 (Busca Complementar):** Após encontrar as exigências na IT 01, busque nas outras ITs (ex: IT 18 para Iluminação, IT 21 para Extintores) para encontrar detalhes ou especificações que complementem a resposta.

3.  **PRIORIZAR E FILTRAR (ANALISAR):** Se a busca retornar múltiplos documentos ou seções conflitantes, aplique esta regra de desempate: **a fonte que corresponder mais precisamente às coordenadas de Área e Altura da pergunta é a correta.** Ignore as outras.
    * *Exemplo Crítico:* Se a pergunta for sobre "250m²" e a busca retornar a "Tabela 5 (área <= 750m²)" e a "Tabela 6 (área > 750m²)", você **DEVE** ignorar a Tabela 6 e basear sua resposta **EXCLUSIVAMENTE** na Tabela 5.

4.  **SINTETIZAR A RESPOSTA (RESPONDER):** Com base nas informações corretas e priorizadas, construa uma resposta completa.
    * Comece com a resposta direta.
    * Detalhe cada exigência encontrada, incorporando as notas de rodapé diretamente na explicação.
    * Ao final, crie uma seção "Fundamentação" e liste todas as fontes que você utilizou. O nome do arquivo ".md" é uma excelente citação.

5.  **PLANO B (FALLBACK):** Apenas se, após seguir rigorosamente todos os passos acima, você não encontrar uma correspondência para as coordenadas da pergunta, utilize a resposta padrão de "Não encontrei a informação...".

---

6.  **Mensagem Inicial:**
    - Ao iniciar uma nova conversa, sua primeira mensagem deve ser:
    > "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
*/
`;

let ragRetriever;

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
    const lastUserMessage = history[history.length - 1] || { parts: [] };
    const textQuery = lastUserMessage.parts.find(p => p.text)?.text || '';

    const contextDocs = await ragRetriever.getRelevantDocuments(textQuery);
    const context = contextDocs.map(doc => `Fonte: ${doc.metadata.source || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');

    const fullHistory = [...history];

    if (fullHistory.length > 0) {
        const lastMessageWithContext = fullHistory[fullHistory.length - 1];
        const newParts = [...lastMessageWithContext.parts];
        const instructionPart = {
            text: `
DOCUMENTAÇÃO TÉCNICA RELEVANTE (ITs e CTs):
${context}
---
INSTRUÇÕES DO SISTEMA (SEMPRE SIGA):
${SYSTEM_PROMPT}
---
DÚVIDA DO ANALISTA:
`
        };
        newParts.unshift(instructionPart);
        fullHistory[fullHistory.length - 1] = { ...lastMessageWithContext, parts: newParts };
    } else {
        fullHistory.push({ role: 'user', parts: [{ text: SYSTEM_PROMPT }] });
    }

    const body = {
        contents: fullHistory,
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
        if (data.promptFeedback && data.promptFeedback.blockReason) {
            throw new Error(`Resposta bloqueada por segurança: ${data.promptFeedback.blockReason}`);
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
  ragRetriever = await initializeRAGEngine();
  app.listen(PORT, () => {
    console.log(`Servidor do Assistente Técnico da DAT a rodar na porta ${PORT}.`);
  });
}

startServer();



