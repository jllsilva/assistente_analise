import fs from 'fs/promises';
import path from 'path';

// Função auxiliar para determinar a tabela correta baseada em área e altura
function getTabelaExigencias(knowledgeBase, grupo, area, altura) {
  if (area <= 750 && altura <= 12) {
    const tabela5 = knowledgeBase.tabela_5_exigencias_area_menor_igual_750;
    const regra = tabela5.regras.find(r => r.criterio.includes(grupo));
    return { ...regra, tabela_info: tabela5.tabela_info };
  } else {
    for (const key in knowledgeBase.tabelas_6_exigencias_area_maior_750) {
      const tabela6 = knowledgeBase.tabelas_6_exigencias_area_maior_750[key];
      if (tabela6.grupo_ocupacao.includes(grupo)) {
        const regra = tabela6.regras_por_altura.find(r => {
          // Lógica simples para encontrar a faixa de altura (pode ser refinada)
          if (altura <= 6 && r.altura.includes("≤ 6")) return true;
          if (altura > 6 && altura <= 12 && r.altura.includes("6 < H ≤ 12")) return true;
          if (altura > 12 && altura <= 23 && r.altura.includes("12 < H ≤ 23")) return true;
          if (altura > 23 && altura <= 30 && r.altura.includes("23 < H ≤ 30")) return true;
          if (altura > 30 && r.altura.includes("Acima de 30")) return true;
          return false;
        });
        return { ...regra, tabela_info: tabela6.tabela_info };
      }
    }
  }
  return null;
}

export async function initializeRAGEngine() {
  try {
    console.log('[RAG Engine] Carregando base de conhecimento JSON...');

    const jsonPath = path.resolve(process.cwd(), 'knowledge_base.json');
    const jsonData = await fs.readFile(jsonPath, 'utf-8');
    const knowledgeBase = JSON.parse(jsonData);

    console.log(`[RAG Engine] Base de conhecimento carregada.`);

    const jsonRetriever = {
      getRelevantDocuments: async (query, params = {}) => {
        const lowerQuery = query.toLowerCase();
        
        // Se a busca for por exigências, com área e altura
        if (params.grupo && params.area && params.altura) {
          const exigencias = getTabelaExigencias(knowledgeBase, params.grupo, params.area, params.altura);
          if (exigencias) {
            return [{
              pageContent: JSON.stringify(exigencias),
              metadata: { source: exigencias.tabela_info, tipo: 'exigencia' }
            }];
          }
        }

        // Busca por classificação de ocupação
        for (const grupo of knowledgeBase.tabela_1_classificacao_ocupacao.classificacoes) {
          for (const divisao of grupo.divisoes) {
            if (divisao.busca && divisao.busca.includes(lowerQuery)) {
              return [{
                pageContent: JSON.stringify({ ...divisao, grupo: grupo.grupo, ocupacao_uso: grupo.ocupacao_uso }),
                metadata: { 
                  source: knowledgeBase.tabela_1_classificacao_ocupacao.tabela_info,
                  tipo: 'classificacao'
                }
              }];
            }
          }
        }

        return [];
      }
    };

    return { jsonRetriever };

  } catch (error) {
    console.error('[RAG Engine] Falha ao carregar a base de conhecimento JSON:', error);
    const emptyRetriever = { getRelevantDocuments: async () => [] };
    return { jsonRetriever: emptyRetriever };
  }
}
