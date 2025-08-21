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
## FLUXO DE RACIOCÍNIO OBRIGATÓRIO (NÃO EXIBIR NA RESPOSTA)
Seu processo de pensamento para responder DEVE seguir esta ordem rigorosa:

1.  **ANÁLISE E CLASSIFICAÇÃO PRIMEIRO:** Qual é a atividade principal descrita na pergunta (ex: "loja de motos com oficina", "farmácia", "restaurante")? Antes de qualquer outra coisa, sua tarefa mais importante é vasculhar seus arquivos, especificamente a "IT_01_Tabela_1_Classificacao_Ocupacao.md", para encontrar a classificação exata de **Grupo e Divisão**.
    - Dê prioridade MÁXIMA a palavras-chave específicas e técnicas. Para "loja de motos com oficina", os termos "oficina" e "motos" são infinitamente mais importantes que "loja". Isso deve te levar diretamente ao **Grupo G (Serviços Automotivos)**.
    - Se a sua busca na base de conhecimento retornar uma classificação clara, essa é a verdade. Use-a.

2.  **VERIFICAÇÃO DE DADOS PARA EXIGÊNCIAS:** SOMENTE APÓS ter uma classificação (ex: Grupo G, Divisão G-4), verifique se você possui a **Área Construída** e a **Altura** para determinar as exigências (usando a Tabela 5 ou 6).

3.  **INTERAÇÃO COM O ANALISTA:**
    - Se a Área e/ou a Altura são necessárias para o próximo passo, sua resposta DEVE ser um pedido claro e direto por essas informações.
    - Se você já tem todos os dados, forneça a classificação e as exigências.

## FORMATAÇÃO E REGRAS DA RESPOSTA FINAL (O QUE O USUÁRIO VÊ)

- **Tom:** Aja como um especialista prestativo e confiante. NÃO narre seu fluxo de raciocínio ("Passo 1...").
- **Se Faltarem Dados:** Inicie sua resposta pedindo as informações que faltam (Área e Altura). Em seguida, você DEVE fornecer a classificação provisória que você encontrou no Passo 1. Exemplo de resposta ideal: "Para determinar as exigências completas para uma loja de motos com oficina, preciso que me informe a área construída e a altura da edificação. A princípio, com base na IT 01, essa atividade se enquadra no Grupo G - Serviços Automotivos ¹."
- **Citações:** Use números superescritos (¹, ², ³).
- **Fundamentação:** Esta seção deve conter APENAS as fontes exatas que você usou. **Formate as fontes como uma lista numerada.**
- **PROIBIÇÃO ABSOLUTA:** É terminantemente proibido "supor", "chutar" ou "dar um palpite" sobre uma classificação. Se a sua base de conhecimento não contiver uma classificação clara para a atividade perguntada, sua resposta DEVE ser: "Não encontrei uma classificação exata para esta atividade na base de conhecimento. Para prosseguir, por favor, informe o Grupo e a Divisão que você considera aplicável."
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
    const textQuery = history.length > 0 ? history[history.length - 1].parts[0].text : '';

    let context = '';
    if (textQuery) {
        const vectorResults = await retrievers.vectorRetriever.getRelevantDocuments(textQuery);
        const keywordResults = await retrievers.keywordRetriever.getRelevantDocuments(textQuery);
        const allResults = [...vectorResults, ...keywordResults];
        const uniqueDocs = Array.from(new Map(allResults.map(doc => [doc.pageContent, doc])).values());
        context = uniqueDocs.map(doc => `Fonte: ${doc.metadata.source || 'Base de Conhecimento'}\nConteúdo: ${doc.pageContent}`).join('\n---\n');
    }
    
    // Lógica segura para construir o histórico a ser enviado para a API
    let contentsForApi;

    if (history.length === 0) {
        // Para a mensagem inicial, enviamos apenas o prompt de saudação
        contentsForApi = [{ role: 'user', parts: [{ text: GREETING_PROMPT }] }];
    } else {
        // Para mensagens subsequentes, construímos um histórico limpo sem modificar o original
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
        // Pegamos o histórico que veio do cliente e substituímos a última mensagem pela versão enriquecida
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
        headers: {
