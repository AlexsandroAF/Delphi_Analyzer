# Relatório de Análise de Código Delphi
Data: {{ timestamp | datetime }}

## Resumo Executivo
Total de arquivos analisados: {{ total_files }}
Total de problemas encontrados: {{ metrics.total_issues }}
Média de problemas por arquivo: {{ "%.2f" | format(metrics.avg_issues_per_file) }}
Arquivos com problemas críticos: {{ metrics.files_with_critical_issues }}

## Distribuição de Problemas por Severidade
{% for severity, count in issues_by_severity.items() %}
- Severidade {{ severity }}: {{ count }} problemas
{% endfor %}

## Tipos de Problemas
{% for type, count in issues_by_type.items() %}
- {{ type }}: {{ count }} ocorrências
{% endfor %}

{% if dependency_metrics %}
## Análise de Dependências
- Total de units: {{ dependency_metrics.total_units }}
- Total de dependências: {{ dependency_metrics.total_dependencies }}
- Dependências circulares: {{ dependency_metrics.circular_dependencies }}
- Units isoladas: {{ dependency_metrics.isolated_units }}
- Unit mais dependente: {{ dependency_metrics.most_dependent }}
- Unit mais referenciada: {{ dependency_metrics.most_depended_on }}
{% endif %}

## Detalhes por Arquivo
{% for file in files_with_issues %}
### {{ file.file_path }}
Prioridade: {{ file.priority }}

Métricas:
{% for key, value in file.metrics.items() %}
- {{ key }}: {{ value }}
{% endfor %}

Problemas encontrados:
{% for issue in file.issues %}
#### {{ issue.type }} (Severidade {{ issue.severity }})
- Descrição: {{ issue.description }}
- Localização: {{ issue.location }}
{% if issue.suggested_fix %}
- Sugestão de correção:
```delphi
{{ issue.suggested_fix }}
```
{% endif %}
{% endfor %}

{% endfor %}

## Recomendações
{% if metrics.files_with_critical_issues > 0 %}
- **URGENTE**: Existem {{ metrics.files_with_critical_issues }} arquivos com problemas críticos que devem ser corrigidos imediatamente.
{% endif %}
{% if issues_by_severity[4] > 0 or issues_by_severity[5] > 0 %}
- **Alta Prioridade**: Corrija os {{ issues_by_severity[4] + issues_by_severity[5] }} problemas de severidade alta (4-5).
{% endif %}
{% if issues_by_severity[3] > 0 %}
- **Média Prioridade**: Planeje a correção dos {{ issues_by_severity[3] }} problemas de severidade média.
{% endif %}
{% if dependency_metrics and dependency_metrics.circular_dependencies > 0 %}
- **Arquitetura**: Revise as {{ dependency_metrics.circular_dependencies }} dependências circulares identificadas.
{% endif %}

## Métricas de API
- Custo total: ${{ "%.2f" | format(metrics.api_cost) }}
- Total de tokens: {{ metrics.api_tokens }}
- Média de tokens por análise: {{ "%.2f" | format(metrics.api_tokens / total_files) }}