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

let vectorStore;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/generate', async (req, res) => {
  const { history } = req.body;
  if (!history) {
    return res.status(400).json({ error: 'O histórico da conversa é obrigatório.' });
  }

  // --- INÍCIO DA CORREÇÃO ---
  // Se for a primeira mensagem (histórico vazio), retorna a saudação diretamente.
  if (history.length === 0) {
    const initialMessage = "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos.";
    return res.json({ reply: initialMessage });
  }
  // --- FIM DA CORREÇÃO ---

  try {
    let context = "";

    if (vectorStore) {
        const textQuery = history[history.length - 1]?.parts[0]?.text || '';
        const retriever = vectorStore.asRetriever();
        const contextDocs = await retriever.getRelevantDocuments(textQuery);
        context = contextDocs.map(doc => `Nome do Arquivo Fonte: ${doc.metadata.source?.split(/[\\/]/).pop() || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    }
    
    const contents = JSON.parse(JSON.stringify(history));

    const body = {
      contents: contents,
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] }
    };

    if (body.contents.length > 0) {
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
