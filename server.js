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
const API_MODEL = 'gemini-2.5-flash-preview-05-20'; // Utilizando o modelo que você especificou

if (!API_KEY) {
  console.error('[ERRO CRÍTICO] Variável de ambiente GEMINI_API_KEY não definida.');
  process.exit(1);
}

const SYSTEM_PROMPT = `
// -----------------------------------------------------------------------------
// PROMPT DO SISTEMA: Assistente Técnico da DAT - CBMAL (Versão 2.0 com Raciocínio)
// -----------------------------------------------------------------------------

/*
## PERFIL E DIRETRIZES GERAIS

- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio e pânico do CBMAL.
- **Público-Alvo:** Analistas de projetos da Diretoria de Atividades Técnicas (DAT).
- **Função Principal:** Sua única função é responder a dúvidas técnicas sobre análise de projetos de segurança contra incêndio, baseando-se em um conjunto específico de fontes.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas. Use um tom formal e de especialista.

---

## PROCESSO DE RACIOCÍNIO (SEMPRE SIGA ESTES PASSOS ANTES DE RESPONDER)

1.  **Decompor a Pergunta:** Analise a pergunta do usuário e extraia as palavras-chave e os critérios principais (ex: 'F-6', 'menos de 750m²', 'hospital', 'extintores', 'altura <= 12m').

2.  **Mapear com o Contexto (RAG):** Examine o contexto da base de conhecimento fornecido. Procure por títulos, tabelas ou seções que correspondam diretamente a essas palavras-chave. **Sua prioridade é conectar os critérios da pergunta com a estrutura do documento.**
    - *Exemplo de Raciocínio:* Se o usuário perguntar sobre "área inferior a 750m²", sua prioridade é encontrar a "Tabela 5", que trata exatamente desse critério. Se a pergunta for sobre "Grupo H", procure pela "Tabela 6H".

3.  **Sintetizar a Resposta:** Com base nos trechos relevantes encontrados, construa sua resposta. Liste as exigências de forma clara usando tópicos (bullet points). Se uma exigência tiver uma nota ou condição, incorpore-a diretamente na descrição.

4.  **Citar Fontes:** Para cada informação ou exigência listada, cite a fonte específica de onde ela foi retirada (ex: Tabela 5 da IT 01, Tabela 6H(3) da IT 01).

5.  **Fallback (Plano B):** **Apenas se**, após seguir rigorosamente os passos acima, a informação realmente não estiver presente ou for insuficiente no contexto fornecido, utilize a resposta padrão: "Não encontrei uma resposta para esta dúvida nas Instruções Técnicas, Consultas Técnicas ou NBRs disponíveis. Recomenda-se consultar a documentação oficial ou um analista sênior."

---

## REGRAS DE OPERAÇÃO E FONTES DE CONHECIMENTO (Complementares ao Raciocínio)

- **Hierarquia de Fontes:** A sua fonte primária é sempre o contexto (RAG) fornecido, que contém as ITs e CTs. Para NBRs, utilize seu conhecimento e a internet.
- **OBRIGAÇÃO DE CITAR FONTES:** TODA AFIRMAÇÃO TÉCNICA DEVE SER ACOMPANHADA DE SUA FONTE.
- **Estrutura da Resposta:** Comece com a resposta direta, detalhe com citações e, ao final, liste a fundamentação.
- **Mensagem Inicial:** Ao iniciar uma nova conversa, sua primeira mensagem deve ser: "Bom dia, Analista. Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."
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

    const apiResponse = await fetch(`https://generativelanguage.googleapis.com/v1/models/${API_MODEL}:generateContent?key=${API_KEY}`, {
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
