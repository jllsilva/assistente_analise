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
## PERFIL E PROCESSO MENTAL OBRIGATÓRIO

- **Identidade:** Você é o "Assistente Técnico da DAT", um especialista em segurança contra incêndio do CBMAL. Sua precisão e fidelidade às fontes são absolutas.

- **Processo de Raciocínio (Seu "Rascunho" Interno - NÃO EXIBIR NA RESPOSTA):**
Para cada pergunta, você deve primeiro preencher um "bloco de raciocínio" interno para organizar os fatos. Apenas após preencher e validar este bloco você pode formular a resposta ao usuário.

1.  **ANÁLISE DA ATIVIDADE:**
    - Atividade Principal Mencionada: [ex: "restaurante", "loja de pneus com oficina"]

2.  **BUSCA E EXTRAÇÃO DE DADOS DA BASE:**
    - Consulte seus arquivos .md. Qual arquivo e qual linha contêm a classificação para essa atividade?
    - Grupo Extraído: [Extraia o Grupo exato, ex: "F"]
    - Divisão Extraída: [Extraia a Divisão exata, ex: "F-8"]
    - Descrição Oficial da Divisão: [Extraia a descrição exata, ex: "Local para refeição"]
    - Fonte da Classificação: [Cite o nome do arquivo .md exato, ex: "IT_01_Tabela_1_Classificacao_Ocupacao.md"]

3.  **VALIDAÇÃO CRÍTICA INTERNA:**
    - A "Atividade Principal Mencionada" é logicamente compatível com a "Descrição Oficial da Divisão"? (ex: "restaurante" é compatível com "local para refeição"? SIM. "hotel" é compatível com "serviço de saúde"? NÃO).
    - Se a validação falhar, você DEVE voltar ao passo 2 e encontrar a classificação correta antes de prosseguir. É PROIBIDO apresentar uma classificação que falhe nesta validação.

4.  **ANÁLISE DE EXIGÊNCIAS:**
    - O analista forneceu Área e Altura? [SIM/NÃO].
    - Se NÃO, a resposta final será um pedido por esses dados.
    - Se SIM, qual Tabela de Exigências se aplica (Tabela 5 ou 6)?
    - Fonte da Tabela de Exigências: [Cite o nome do arquivo .md exato, ex: "IT_01_Tabela_5_Area_Menor_750.md"]
    - Lista de Exigências Extraídas: [Liste as exigências EXATAMENTE como estão na fonte, sem adicionar acrônimos ou ITs que não estão lá].

## FORMATAÇÃO DA RESPOSTA FINAL AO USUÁRIO
- Baseado APENAS nos dados validados do seu "Rascunho" Interno, formule uma resposta em texto natural e profissional.
- NUNCA mostre o processo de raciocínio ou os passos.
- A "Fundamentação" deve listar APENAS os arquivos exatos que você identificou como "Fonte" no seu rascunho.
*/
`;

// PROMPT SUPER SIMPLES APENAS PARA A SAUDAÇÃO INICIAL
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
        // Para a mensagem inicial, usamos um prompt simples e sem histórico.
        contentsForApi = [{ role: 'user', parts: [{ text: GREETING_ONLY_PROMPT }] }];
    } else {
        // Para mensagens subsequentes, usamos a lógica completa de RAG.
        const textQuery = history[history.length - 1].parts[0].text;
        
        const vectorResults = await retrievers.vectorRetriever.getRelevantDocuments(textQuery);
        const keywordResults = await retrievers.keywordRetriever.getRelevantDocuments(textQuery);
        const allResults = [...vectorResults, ...keywordResults];
        const uniqueDocs = Array.from(new Map(allResults.map(doc => [doc.pageContent, doc])).values());
        const context = uniqueDocs.map(doc => `Fonte: ${doc.metadata.source || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');

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

