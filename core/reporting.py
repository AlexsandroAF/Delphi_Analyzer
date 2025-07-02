# reporting.py
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.database import CodeAnalysisDatabase
from core.models import AnalysisResult


class ReportGenerator:
    """Gerador de relatórios com suporte a markdown e html"""

    def __init__(self, template_dir: str = "templates"):
        self.logger = logging.getLogger(__name__)
        self.template_dir = Path(template_dir)
        self._setup_jinja()
        self._setup_plotting()

    def _setup_jinja(self):
        """Configura ambiente Jinja2 para templates"""
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Adiciona filtros personalizados
        self.jinja_env.filters['datetime'] = lambda dt: dt.strftime('%Y-%m-%d %H:%M:%S')
        self.jinja_env.filters['severity_color'] = self._get_severity_color

    def _setup_plotting(self):
        """Configura estilo para gráficos"""
        # Usando estilo padrão do matplotlib em vez do seaborn
        plt.style.use('default')

        # Configurações básicas de estilo
        mpl.rcParams.update({
            'figure.figsize': [10, 6],
            'axes.grid': True,
            'grid.linestyle': ':',
            'grid.alpha': 0.5,
            'axes.facecolor': '#f0f0f0',
            'figure.facecolor': 'white',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14
        })

    def _get_severity_color(self, severity: int) -> str:
        """Retorna cor baseada na severidade"""
        colors = {
            1: '#4caf50',  # Verde
            2: '#8bc34a',  # Verde claro
            3: '#ffc107',  # Amarelo
            4: '#ff9800',  # Laranja
            5: '#f44336'  # Vermelho
        }
        return colors.get(severity, '#9e9e9e')

    async def generate_analysis_report(self,
                                       analysis_results: List[AnalysisResult],
                                       dep_graph: Optional[nx.DiGraph] = None,
                                       output_dir: Path = Path("../reports"),
                                       formats: List[str] = ['html', 'md']) -> Dict[str, Path]:
        """Gera relatório de análise em HTML e/ou Markdown"""
        try:
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepara dados para o template
            template_data = self._prepare_template_data(analysis_results, dep_graph)

            # Gera gráficos
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            plot_paths = self._generate_plots(template_data, plots_dir)
            template_data['plots'] = plot_paths

            # Gera relatórios nos formatos solicitados
            outputs = {}

            if 'md' in formats:
                md_path = output_dir / f"analysis_report_{timestamp}.md"
                template = self.jinja_env.get_template("analysis_report.md")
                content = template.render(**template_data)
                md_path.write_text(content, encoding='utf-8')
                outputs['md'] = md_path
                self.logger.info(f"Relatório Markdown gerado: {md_path}")

            if 'html' in formats:
                html_path = output_dir / f"analysis_report_{timestamp}.html"
                template = self.jinja_env.get_template("analysis_report.html")
                content = template.render(**template_data)
                html_path.write_text(content, encoding='utf-8')
                outputs['html'] = html_path
                self.logger.info(f"Relatório HTML gerado: {html_path}")

            return outputs

        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório: {str(e)}")
            raise


    def _prepare_template_data(self,
                               results: List[AnalysisResult],
                               dep_graph: Optional[nx.DiGraph]) -> Dict[str, Any]:
        """Prepara dados para os templates"""
        data = {
            'timestamp': datetime.now(),
            'total_files': len(results),
            'issues_by_severity': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'issues_by_type': {},
            'files_with_issues': [],
            'metrics': {
                'total_issues': 0,
                'avg_issues_per_file': 0,
                'files_with_critical_issues': 0
            }
        }

        # Processa resultados
        for result in results:
            for issue in result.issues:
                severity = issue.get('severity', 1)
                issue_type = issue.get('type', 'unknown')

                data['issues_by_severity'][severity] = \
                    data['issues_by_severity'].get(severity, 0) + 1
                data['issues_by_type'][issue_type] = \
                    data['issues_by_type'].get(issue_type, 0) + 1

            if result.issues:
                data['files_with_issues'].append({
                    'file_path': result.file_path,
                    'issues': result.issues,
                    'metrics': result.metrics,
                    'priority': result.priority
                })

        # Calcula métricas
        data['metrics']['total_issues'] = sum(data['issues_by_severity'].values())
        data['metrics']['avg_issues_per_file'] = \
            data['metrics']['total_issues'] / len(results) if results else 0
        data['metrics']['files_with_critical_issues'] = sum(
            1 for r in results
            if any(i.get('severity', 1) >= 4 for i in r.issues)
        )

        # Adiciona métricas de dependência se disponível
        if dep_graph:
            data['dependency_metrics'] = {
                'total_units': len(dep_graph.nodes),
                'total_dependencies': len(dep_graph.edges),
                'circular_dependencies': len(list(nx.simple_cycles(dep_graph))),
                'isolated_units': len(list(nx.isolates(dep_graph))),
                'most_dependent': max(
                    dep_graph.nodes,
                    key=lambda n: dep_graph.out_degree(n),
                    default=None
                ),
                'most_depended_on': max(
                    dep_graph.nodes,
                    key=lambda n: dep_graph.in_degree(n),
                    default=None
                )
            }

        return data

    def _generate_plots(self, data: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """Gera gráficos para o relatório"""
        plots = {}

        # Gráfico de severidade
        plt.figure(figsize=(10, 6))
        severities = list(data['issues_by_severity'].keys())
        counts = list(data['issues_by_severity'].values())
        colors = [self._get_severity_color(s) for s in severities]

        plt.bar(severities, counts, color=colors)
        plt.title('Distribuição de Issues por Severidade')
        plt.xlabel('Nível de Severidade')
        plt.ylabel('Número de Issues')

        severity_plot = output_dir / "severity_distribution.png"
        plt.savefig(severity_plot)
        plt.close()
        plots['severity'] = severity_plot

        # Gráfico de tipos de issues
        plt.figure(figsize=(12, 6))
        issue_types = list(data['issues_by_type'].keys())
        type_counts = list(data['issues_by_type'].values())

        plt.barh(issue_types, type_counts)
        plt.title('Issues por Tipo')
        plt.xlabel('Número de Ocorrências')

        types_plot = output_dir / "issues_by_type.png"
        plt.savefig(types_plot)
        plt.close()
        plots['types'] = types_plot

        # Se houver métricas de dependência
        if 'dependency_metrics' in data:
            # Gráfico de dependências
            plt.figure(figsize=(10, 6))
            metrics = data['dependency_metrics']
            metric_names = ['Units', 'Dependencies', 'Circular', 'Isolated']
            metric_values = [
                metrics['total_units'],
                metrics['total_dependencies'],
                metrics['circular_dependencies'],
                metrics['isolated_units']
            ]

            plt.bar(metric_names, metric_values)
            plt.title('Métricas de Dependência')
            plt.xticks(rotation=45)

            dep_plot = output_dir / "dependency_metrics.png"
            plt.savefig(dep_plot)
            plt.close()
            plots['dependencies'] = dep_plot

        return plots

    async def generate_metrics_report(self,
                                      db: CodeAnalysisDatabase,
                                      output_path: Path):
        """Gera relatório de métricas do projeto"""
        try:
            # Obtém métricas do banco
            project_metrics = await db.get_project_metrics()

            report_data = {
                'timestamp': datetime.now().isoformat(),
                'project_metrics': project_metrics,
                'analysis_history': {
                    'total_analyses': project_metrics.get('total_files_analyzed', 0),
                    'total_issues_found': sum(
                        project_metrics.get('issues_by_priority', {}).values()
                    ),
                    'total_fixes_applied': project_metrics.get('applied_changes', 0)
                },
                'api_usage': {
                    'total_cost': project_metrics.get('api_total_cost', 0),
                    'total_tokens': project_metrics.get('api_total_tokens', 0)
                }
            }

            # Gera relatório em YAML
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(report_data, f, default_flow_style=False)

        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório de métricas: {str(e)}")
            raise

    def export_issues_csv(self,
                          results: List[AnalysisResult],
                          output_path: Path):
        """Exporta issues para CSV"""
        try:
            issues_data = []

            for result in results:
                for issue in result.issues:
                    issues_data.append({
                        'file': result.file_path,
                        'type': issue.get('type', 'unknown'),
                        'severity': issue.get('severity', 1),
                        'description': issue.get('description', ''),
                        'location': issue.get('location', ''),
                        'suggested_fix': issue.get('suggested_fix', '')
                    })

            df = pd.DataFrame(issues_data)
            df.to_csv(output_path, index=False)

        except Exception as e:
            self.logger.error(f"Erro ao exportar issues: {str(e)}")
            raise