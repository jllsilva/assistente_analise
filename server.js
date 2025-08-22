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

// Prompts
const GREETING_ONLY_PROMPT = `Sua única tarefa é responder com a seguinte frase, e nada mais: "Saudações, Sou o Assistente Técnico da DAT. Estou à disposição para responder suas dúvidas sobre as Instruções Técnicas, Consultas Técnicas e NBRs aplicáveis à análise de projetos."`;

const QUERY_ANALYSIS_PROMPT = `
Analise a pergunta do usuário e extraia as informações em um formato JSON.

Tipos de busca possíveis: "CLASSIFICACAO", "EXIGENCIA", "DEFINICAO", "INDEFINIDO".

- Use "CLASSIFICACAO" para perguntas sobre o grupo ou divisão de uma atividade.
  - Exemplo: "classificar restaurante", "qual o grupo de um shopping?"
  - O campo "termo" deve ser o objeto da classificação.

- Use "EXIGENCIA" para perguntas sobre medidas de segurança, preventivos, ou o que é necessário para uma edificação.
  - Sinônimos para ficar atento: "exigências", "preventivos", "medidas de segurança", "o que precisa ter", "quais os itens".
  - Exemplo: "quais as exigências para F-6 com 750m2 e 6m", "preventivos para H-4", "o que uma borracharia precisa ter?"
  - Extraia os campos "grupo", "divisao", "area" e "altura" se estiverem presentes.

- Use "DEFINICAO" para perguntas conceituais.
  - Exemplo: "o que é CMAR?", "explique compartimentação"
  - O campo "termo" deve ser o conceito.

- Se a intenção não for clara, retorne "tipo_busca": "INDEFINIDO".

Pergunta do Usuário: "{QUERY}"

JSON de Análise:
`;

const RESPONSE_GENERATION_PROMPT = `
/*
## PERFIL
- Você é o "Assistente Técnico da DAT". Sua precisão e fidelidade às fontes são absolutas.

## CONTEXTO FORNECIDO
{CONTEXT}

## TAREFA
- Baseado EXCLUSIVAMENTE no CONTEXTO FORNECIDO, responda à DÚVIDA DO ANALISTA.
- Se o contexto for uma ficha JSON de classificação, apresente os dados e peça as informações que faltam (área/altura).
- Se o contexto for uma ficha JSON de exigências, liste as exigências de forma clara.
- Se o contexto for um texto, use-o para responder a pergunta conceitual.
- Se o contexto estiver vazio, responda que não encontrou informações na base de conhecimento.
- Formate a resposta de forma profissional, com citações (¹, ²) e a seção "Fundamentação" no final, usando o campo "source" ou "tabela_info" do contexto.

## DÚVIDA DO ANALISTA
{QUERY}
*/
`;

let ragTools;
let conversationState = {};

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
    if (history.length === 0) {
        conversationState = {}; // Limpa o estado para uma nova conversa
        const reply = await callGemini(GREETING_ONLY_PROMPT);
        return res.json({ reply });
    }

    const textQuery = history[history.length - 1].parts[0].text;
    
    // FASE 1: Análise da Pergunta
    const analysisPrompt = QUERY_ANALYSIS_PROMPT.replace('{QUERY}', textQuery);
    let analysisResultText = await callGemini(analysisPrompt);
    analysisResultText = analysisResultText.replace(/```json|```/g, '').trim();
    const analysis = JSON.parse(analysisResultText);

    let contextDocs = [];
    
    // FASE 2: Busca Direcionada
if (analysis.tipo_busca === 'CLASSIFICACAO' && analysis.termo) {
        const classificacoes = ragTools.knowledgeBaseJSON.tabela_1_classificacao_ocupacao.classificacoes;
        
        // 1. Divide o termo de busca da IA em palavras-chave individuais.
        const palavrasChave = analysis.termo.toLowerCase().split(/\s+/);

        for (const grupo of classificacoes) {
            for (const divisao of grupo.divisoes) {
                if (divisao.busca) {
                    // 2. Verifica se ALGUMA das palavras-chave existe no campo "busca" do JSON.
                    const encontrou = palavrasChave.some(palavra => divisao.busca.includes(palavra));
                    if (encontrou) {
                        contextDocs.push({
                            pageContent: JSON.stringify({ ...divisao, grupo: grupo.grupo, ocupacao_uso: grupo.ocupacao_uso }),
                            metadata: { source: ragTools.knowledgeBaseJSON.tabela_1_classificacao_ocupacao.tabela_info }
                        });
                        conversationState = { lastClassification: { grupo: grupo.grupo, divisao: divisao.divisao } };
                        break; 
                    }
                }
            }
            if (contextDocs.length > 0) break;
        }
    }
    } else if (analysis.tipo_busca === 'EXIGENCIA') {
        const grupo = analysis.grupo || conversationState.lastClassification?.grupo;
        const area = analysis.area;
        const altura = analysis.altura;

        if (grupo && area && altura) {
            const exigencias = ragTools.getTabelaExigencias(ragTools.knowledgeBaseJSON, grupo.charAt(0), area, altura);
            if (exigencias) {
                contextDocs.push({
                    pageContent: JSON.stringify(exigencias),
                    metadata: { source: exigencias.tabela_info }
                });
            }
            conversationState = {}; // Limpa o estado após a consulta completa
        } else {
            // Se faltar dados, a IA pedirá mais informações
            contextDocs = []; 
        }

    } else { // Para DEFINICAO ou INDEFINIDO, busca no texto
        contextDocs = await ragTools.textRetriever.getRelevantDocuments(textQuery);
    }
    
    const context = contextDocs.map(doc => `Fonte: ${doc.metadata.source}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    
    // FASE 3: Geração da Resposta
    const finalPrompt = RESPONSE_GENERATION_PROMPT
        .replace('{CONTEXT}', context)
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
    console.log(`Servidor do Assistente Técnico da DAT a rodar na porta ${PORT}.`);
  });
}

startServer();

