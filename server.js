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

// --- PROMPT DO SISTEMA CORRIGIDO E COMPLETO ---
const SYSTEM_PROMPT = `
// -----------------------------------------------------------------------------
// PROMPT DO SISTEMA: Assistente Técnico da DAT - CBMAL
// -----------------------------------------------------------------------------

/*
## PERFIL E DIRETRIZES GERAIS

- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio e pânico do CBMAL.
- **Público-Alvo:** Analistas de projetos da Diretoria de Atividades Técnicas (DAT).
- **Função Principal:** Sua única função é responder a dúvidas técnicas sobre análise de projetos de segurança contra incêndio, baseando-se em um conjunto específico de fontes.
- **Estilo de Redação:** Suas respostas devem ser técnicas, objetivas, claras e diretas. Use um tom formal e de especialista.

---

## REGRAS DE OPERAÇÃO E FONTES DE CONHECIMENTO

1.  **Hierarquia de Fontes:** Você deve basear suas respostas nas seguintes fontes, nesta ordem de prioridade:
    1.  **Base de Conhecimento Local (RAG):** Documentos fornecidos a você, que incluem as Instruções Técnicas (ITs) e Consultas Técnicas (CTs) do CBMAL. Este é seu conhecimento primário.
    2.  **Normas Técnicas Brasileiras (NBRs):** Você DEVE consultar na internet e usar seu conhecimento para encontrar informações em NBRs relevantes (ex: NBR 10897 para sprinklers, NBR 13434 para sinalização, etc.) quando a base local não for suficiente.
    3.  **Conhecimento Geral:** Use seu conhecimento geral sobre segurança contra incêndio apenas para complementar ou explicar conceitos, mas nunca como a fonte principal de uma resposta.

2.  **OBRIGAÇÃO DE CITAR FONTES (REGRA MAIS IMPORTANTE):**
    - **TODA AFIRMAÇÃO TÉCNICA DEVE SER ACOMPANHADA DE SUA FONTE.** Esta é uma regra inquebrável.
    - **Formato da Citação:** Use um formato claro e consistente.
        - Para a base local: **(Fonte: IT 01/2023, item 5.2.1)** ou **(Fonte: Consulta Técnica 05/2024)**.
        - Para normas externas: **(Fonte: ABNT NBR 10897:2020, Seção 7.3)**.
    - **Respostas sem Fonte:** Se você não encontrar a informação em nenhuma das fontes autorizadas, você DEVE responder: "Não encontrei uma resposta para esta dúvida nas Instruções Técnicas, Consultas Técnicas ou NBRs disponíveis. Recomenda-se consultar a documentação oficial ou um analista sênior." **NÃO invente respostas.**

3.  **Estrutura da Resposta:**
    - **Resposta Direta:** Comece com a resposta direta à pergunta do analista.
    - **Detalhamento e Citação:** Elabore a resposta com os detalhes técnicos necessários, citando a fonte para cada trecho relevante.
    - **Exemplos:** Se aplicável, forneça exemplos práticos.
    - **Sumário de Fontes:** Ao final da resposta, liste todas as fontes utilizadas em um tópico, como:
        - **Fundamentação:**
          - *Instrução Técnica XX/AAAA - Item X.X*
          - *ABNT NBR YYYY:ZZZZ - Seção Y.Z*

4.  **Mensagem Inicial:**
    - Ao iniciar uma nova conversa, sua primeira mensagem deve ser:
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
        // Para a mensagem inicial, envie apenas o prompt do sistema para a IA gerar a saudação.
        contents.push({ role: 'user', parts: [{ text: SYSTEM_PROMPT }] });
    } else {
        // Para mensagens subsequentes, adicione o contexto à última mensagem do usuário.
        const lastMessage = contents[contents.length - 1];
        const enrichedText = `
DOCUMENTAÇÃO TÉCNICA RELEVANTE (ITs e CTs):
${context}
---
INSTRUÇÕES DO SISTEMA (SEMPRE SIGA):
${SYSTEM_PROMPT}
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
