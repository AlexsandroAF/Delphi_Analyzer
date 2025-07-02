# Dossie do Projeto - Analisador de Código Delphi com IA

## Visão Geral
O projeto é um sistema de análise e correção automática de código Delphi usando inteligência artificial (Claude da Anthropic). Ele analisa o código fonte, identifica problemas, sugere correções e pode aplicá-las automaticamente, com suporte específico para análise baseada em regiões ($REGION) do Delphi.

## Estrutura de Arquivos

### 1. models.py
**Função:** Define as estruturas de dados básicas do sistema
- `DelphiToken`: Representa tokens individuais do código
- `CodeBlock`: Representa blocos de código (procedures, functions, etc)
- `CodeChange`: Registra mudanças feitas no código
- `AnalysisResult`: Armazena resultados de análises
- `ApiInteraction`: Registra interações com a API da IA
- `DelphiFile`: Representa um arquivo Delphi completo
- `RegionBlock`: Representa uma região do código ($REGION)

### 2. database.py
**Função:** Gerencia todo o armazenamento de dados
- Cria e mantém banco SQLite
- Armazena histórico de análises
- Registra mudanças no código
- Mantém cache de análises
- Rastreia uso da API
- Gera métricas e estatísticas
- Suporte a serialização JSON para datetime

### 3. syntax_validator.py
**Função:** Valida a sintaxe do código Delphi
- Verifica estrutura básica das units
- Valida declarações e implementações
- Identifica erros de sintaxe
- Verifica balanceamento de blocos
- Sugere correções sintáticas
- Validação de regiões ($REGION)

### 4. delphi_parser.py
**Função:** Análise estática do código
- Parse completo de código Delphi
- Extrai estruturas e dependências
- Mapeia relacionamentos entre units
- Identifica componentes e suas relações
- Analisa dependências circulares
- Processamento de regiões ($REGION)

### 5. config.py
**Função:** Gerencia configurações do sistema
- Carrega configurações de diferentes fontes
- Gerencia dados sensíveis (API keys)
- Configura diretórios e caminhos
- Define parâmetros de análise
- Controla comportamento do sistema
- Criptografia simplificada para dados sensíveis

### 6. editor.py
**Função:** Manipula arquivos de código
- Aplica mudanças ao código
- Gerencia sistema de backup
- Valida mudanças antes de aplicar
- Permite rollback de alterações
- Mantém histórico de modificações
- Suporte a manipulação de regiões

### 7. analyzer.py
**Função:** Núcleo do sistema de análise
- Integra com a API do Claude
- Analisa código em busca de problemas
- Gera sugestões de melhorias
- Avalia segurança e performance
- Coordena processo de análise
- Controle de rate limit da API
- Otimização de código para análise
- Análise baseada em regiões ($REGION)

### 8. reporting.py
**Função:** Gera relatórios e documentação
- Cria relatórios em HTML e Markdown
- Gera visualizações e gráficos
- Produz métricas e estatísticas
- Documenta mudanças e análises
- Exporta resultados
- Suporte a UTF-8 em relatórios

### 9. main.py
**Função:** Interface principal do sistema
- Fornece CLI para interação
- Coordena execução de análises
- Gerencia fluxo de trabalho
- Mostra progresso e resultados
- Permite diferentes modos de operação
- Suporte a comandos assíncronos

## Funcionalidades Principais

1. **Análise de Código**
   - Análise por regiões ($REGION)
   - Detecção de problemas
   - Verificação de segurança
   - Análise de performance
   - Identificação de bugs
   - Validação de padrões
   - Otimização de código para análise

2. **Correção Automática**
   - Sugestão de correções por região
   - Aplicação automática de fixes
   - Sistema de backup
   - Validação de mudanças
   - Rollback em caso de erro
   - Preservação de contexto das regiões

3. **Gestão de API**
   - Controle de rate limit
   - Otimização de tokens
   - Retry automático
   - Cache de respostas
   - Validação de JSON
   - Tratamento de caracteres UTF-8

4. **Relatórios**
   - Documentação detalhada
   - Visualizações gráficas
   - Métricas e estatísticas
   - Histórico de análises
   - Exportação em HTML/Markdown
   - Suporte a caracteres especiais

## Uso Principal

O sistema é especialmente útil para:
1. Manutenção de código legado
2. Garantia de qualidade
3. Refatoração segura
4. Documentação automática
5. Análise de dependências
6. Análise contextual por regiões

## Requisitos de Código

Para melhor aproveitamento do sistema:
1. Código Delphi organizado com $REGION
2. Regiões bem definidas e nomeadas
3. Estrutura consistente de units
4. Uso adequado de tratamento de exceções
5. Documentação básica do código