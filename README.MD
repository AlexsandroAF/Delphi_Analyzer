# Configuração do Sistema

Este documento descreve a estrutura de configuração do analisador de código Delphi.

## Arquivos de Configuração

- `config.yml`: Configurações padrão do projeto
- `config.local.yml`: Configurações específicas da máquina local (não versionado)
- `.env`: Variáveis de ambiente
- `.env.key`: Chave de criptografia para dados sensíveis (gerada automaticamente)

## Estrutura de Configuração

### Database
- `path`: Caminho do banco de dados SQLite
- `backup_retention_days`: Dias para manter backups
- `cache_enabled`: Habilita/desabilita cache
- `cache_ttl_days`: Tempo de vida do cache em dias

### API
- `key`: Chave da API Anthropic (criptografada)
- `model`: Modelo do Claude a ser usado
- `max_tokens`: Limite de tokens por requisição
- `temperature`: Temperatura para geração de texto
- `timeout`: Timeout da API em segundos
- `max_retries`: Número máximo de tentativas

### Analysis
- `min_severity`: Severidade mínima para reportar
- `auto_fix_enabled`: Habilita correções automáticas
- `auto_fix_min_severity`: Severidade mínima para correção automática
- `monitor_interval`: Intervalo de monitoramento em segundos
- `ignore_patterns`: Padrões de arquivo a ignorar

### Diretórios
- `project_dir`: Diretório do projeto Delphi
- `backup_dir`: Diretório para backups
- `log_dir`: Diretório para logs

## Configuração Inicial

1. Copie `config.yml` para `config.local.yml`
2. Edite `config.local.yml` com suas configurações locais
3. Crie um arquivo `.env` baseado no exemplo
4. Execute o sistema - a chave de criptografia será gerada automaticamente

## Segurança

- Nunca comite `config.local.yml`, `.env` ou `.env.key`
- A chave da API é automaticamente criptografada
- Use variáveis de ambiente para configurações sensíveis em produção

## Exemplo de Uso

```python
from core.config import ConfigManager

# Carrega configuração
config_manager = ConfigManager()
config = config_manager.load_config()

# Acessa configurações
db_path = config.database.path
api_key = config.api.key
```

## Variáveis de Ambiente

- `ANTHROPIC_API_KEY`: Chave da API Anthropic
- `DELPHI_PROJECT_DIR`: Diretório do projeto
- `DEBUG`: Modo debug (true/false)
- `ENVIRONMENT`: Ambiente (development/production)

As variáveis de ambiente têm precedência sobre os arquivos de configuração.