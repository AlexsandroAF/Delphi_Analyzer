import asyncio
import logging
import signal
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.analyzer import DelphiAnalyzer
from core.config import ConfigManager, SystemConfig
from core.database import CodeAnalysisDatabase
from core.delphi_parser import DelphiParser
from core.editor import DelphiCodeEditor
from core.reporting import ReportGenerator

console = Console()

def handle_shutdown(signum, frame):
    """Manipulador de sinal para shutdown limpo"""
    console.print("\n[yellow]Encerrando sistema...[/]")
    try:
        loop = asyncio.get_event_loop()
        if 'db' in globals() and db:
            loop.run_until_complete(db.close())
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

def setup_basic_logging():
    """Configura logging básico para o sistema"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('delphi_analyzer.log', encoding='utf-8')
        ]
    )

def run_async(func):
    """Decorator para rodar funções assíncronas no Click"""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

def setup_logging(config: SystemConfig):
    """Configura sistema de logging"""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'delphi_analyzer.log'),
            logging.StreamHandler()
        ]
    )


async def initialize_system(config_path: str) -> tuple:
    """Inicializa todos os componentes do sistema"""
    db = None
    try:
        # Carrega configuração
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()

        # Configura logging
        setup_logging(config)

        # Inicializa componentes
        db = await CodeAnalysisDatabase.create(config.database.path)
        analyzer = DelphiAnalyzer(config.project_dir, config.api.key, db)
        parser = DelphiParser()
        editor = DelphiCodeEditor(config.backup_dir)
        reporter = ReportGenerator()

        return config, db, analyzer, parser, editor, reporter

    except Exception as e:
        if db:
            await db.close()
        raise


@click.group()
@click.option('--config', default='config.yml', help='Caminho do arquivo de configuração')
@click.pass_context
def cli(ctx, config):
    """Analisador de código Delphi com IA"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


# main.py - ajuste apenas na parte do comando analyze
@cli.command()
@click.argument('project_dir', type=click.Path(exists=True))
@click.option('--output', '-o', default='reports', help='Diretório de saída para relatórios')
@click.option('--format', '-f', multiple=True,
              type=click.Choice(['md', 'html'], case_sensitive=False),
              default=['html'], help='Formato(s) do relatório')
@click.option('--analyze-deps/--no-deps', default=True, help='Analisar dependências')
@click.pass_context
@run_async
async def analyze(ctx, project_dir, output, format, analyze_deps):
    """Analisa um projeto Delphi completo"""
    db = None
    try:
        with console.status("[bold green]Inicializando sistema...") as status:
            config, db, analyzer, parser, editor, reporter = await initialize_system(
                ctx.obj['config_path']
            )

            status.update("[bold yellow]Analisando projeto...")
            results = await analyzer.analyze_project()

            if analyze_deps:
                status.update("[bold yellow]Analisando dependências...")
                dep_graph = await parser.analyze_dependencies(Path(project_dir))
            else:
                dep_graph = None

            status.update("[bold yellow]Gerando relatórios...")
            output_path = Path(output)
            output_path.mkdir(exist_ok=True)

            report_files = await reporter.generate_analysis_report(
                results['results'],
                dep_graph,
                output_path,
                list(format)
            )

            # Exibe resumo
            console.print("\n[bold green]Análise concluída com sucesso![/]")

            table = Table(title="Resumo da Análise")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", justify="right")

            stats = results['statistics']
            table.add_row("Arquivos Analisados", str(stats['analyzed_files']))
            table.add_row("Arquivos Corrigidos", str(stats['fixed_files']))
            table.add_row("Erros Encontrados", str(stats['errors']))

            console.print(table)

            # Mostra arquivos gerados
            console.print("\n[bold]Relatórios gerados:[/]")
            for fmt, path in report_files.items():
                console.print(f"  - {fmt.upper()}: {path}")

    except Exception as e:
        console.print(f"\n[bold red]Erro:[/] {str(e)}")
        sys.exit(1)

@cli.command('analyze-project')
@click.argument('project_dir', type=click.Path(exists=True))
@click.option('--output', '-o', default='reports', help='Diretório de saída para relatórios')
@click.option('--format', '-f', multiple=True,
              type=click.Choice(['md', 'html'], case_sensitive=False),
              default=['html'], help='Formato(s) do relatório')
@click.option('--analyze-deps/--no-deps', default=True, help='Analisar dependências')
@run_async
async def analyze_project(project_dir, output, format, analyze_deps):
    """Analisa um projeto Delphi completo"""
    try:
        with console.status("[bold green]Inicializando sistema...") as status:
            config_manager = ConfigManager('config.yml')
            config = config_manager.load_config()
            db = await CodeAnalysisDatabase.create(config.database.path)
            analyzer = DelphiAnalyzer(config.project_dir, config.api.key, db)
            parser = DelphiParser()
            reporter = ReportGenerator()

            status.update("[bold yellow]Analisando projeto...")
            results = await analyzer.analyze_project()

            if analyze_deps:
                status.update("[bold yellow]Analisando dependências...")
                dep_graph = await parser.analyze_dependencies(Path(project_dir))
            else:
                dep_graph = None

            status.update("[bold yellow]Gerando relatórios...")
            output_path = Path(output)
            output_path.mkdir(exist_ok=True)

            report_files = await reporter.generate_analysis_report(
                results['results'],
                dep_graph,
                output_path,
                list(format)
            )

            # Exibe resumo
            console.print("\n[bold green]Análise concluída com sucesso![/]")

            table = Table(title="Resumo da Análise")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", justify="right")

            stats = results['statistics']
            table.add_row("Arquivos Analisados", str(stats['analyzed_files']))
            table.add_row("Arquivos Corrigidos", str(stats['fixed_files']))
            table.add_row("Erros Encontrados", str(stats['errors']))

            console.print(table)

            # Mostra arquivos gerados
            console.print("\n[bold]Relatórios gerados:[/]")
            for fmt, path in report_files.items():
                console.print(f"  - {fmt.upper()}: {path}")

    except Exception as e:
        console.print(f"\n[bold red]Erro:[/] {str(e)}")
        sys.exit(1)

@cli.command('analyze-file')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--fix/--no-fix', default=False, help='Aplicar correções automáticas')
@run_async
async def analyze_single_file(file_path, fix):
    """Analisa um único arquivo Delphi"""
    try:
        config_manager = ConfigManager('config.yml')
        config = config_manager.load_config()
        db = await CodeAnalysisDatabase.create(config.database.path)
        analyzer = DelphiAnalyzer(config.project_dir, config.api.key, db)

        with console.status("[bold yellow]Analisando arquivo...") as status:
            result = await analyzer.analyze_file(Path(file_path))

            if fix and result.issues:
                status.update("[bold yellow]Aplicando correções...")
                await analyzer.apply_suggestions(
                    Path(file_path),
                    result.issues,
                    min_priority=4
                )

            # Mostra resultados
            console.print("\n[bold]Resultados da Análise[/]")

            if not result.issues:
                console.print("[green]Nenhum problema encontrado![/]")
                return

            for issue in result.issues:
                severity = issue['severity']
                color = {
                    1: "green",
                    2: "blue",
                    3: "yellow",
                    4: "orange",
                    5: "red"
                }.get(severity, "white")

                console.print(Panel(
                    f"{issue['description']}\n"
                    f"[bold]Localização:[/] {issue['location']}\n"
                    f"[bold]Tipo:[/] {issue['type']}",
                    title=f"[{color}]Severidade {severity}[/]",
                    expand=False
                ))

    except Exception as e:
        console.print(f"\n[bold red]Erro:[/] {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--fix/--no-fix', default=False, help='Aplicar correções automáticas')
@click.pass_context
@run_async
async def analyze_file(ctx, file_path, fix):
    """Analisa um único arquivo Delphi"""
    try:
        config, db, analyzer, parser, editor, reporter = await initialize_system(
            ctx.obj['config_path']
        )

        with console.status("[bold yellow]Analisando arquivo...") as status:
            result = await analyzer.analyze_file(Path(file_path))

            if fix and result.issues:
                status.update("[bold yellow]Aplicando correções...")
                await analyzer.apply_suggestions(
                    Path(file_path),
                    result.issues,
                    min_priority=4
                )

            # Mostra resultados
            console.print("\n[bold]Resultados da Análise[/]")

            if not result.issues:
                console.print("[green]Nenhum problema encontrado![/]")
                return

            for issue in result.issues:
                severity = issue['severity']
                color = {
                    1: "green",
                    2: "blue",
                    3: "yellow",
                    4: "orange",
                    5: "red"
                }.get(severity, "white")

                console.print(Panel(
                    f"{issue['description']}\n"
                    f"[bold]Localização:[/] {issue['location']}\n"
                    f"[bold]Tipo:[/] {issue['type']}",
                    title=f"[{color}]Severidade {severity}[/]",
                    expand=False
                ))

    except Exception as e:
        console.print(f"\n[bold red]Erro:[/] {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
@run_async
async def monitor(ctx):
    """Monitora continuamente o projeto por mudanças"""
    try:
        config, db, analyzer, parser, editor, reporter = await initialize_system(
            ctx.obj['config_path']
        )

        console.print("[bold green]Iniciando monitoramento...[/]")
        console.print(f"Monitorando diretório: {config.project_dir}")
        console.print("Pressione Ctrl+C para parar")

        await analyzer.monitor_changes(config.analysis.monitor_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoramento interrompido pelo usuário[/]")
    except Exception as e:
        console.print(f"\n[bold red]Erro:[/] {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--days', default=7, help='Número de dias para manter backups')
@click.pass_context
@run_async
async def cleanup(ctx, days):
    """Limpa backups antigos e cache"""
    try:
        config, db, analyzer, parser, editor, reporter = await initialize_system(
            ctx.obj['config_path']
        )

        with console.status("[bold yellow]Limpando arquivos antigos...") as status:
            # Limpa backups
            await editor.cleanup_old_backups(days)

            # Limpa cache
            await db.cleanup_old_cache(days)

            console.print("[bold green]Limpeza concluída com sucesso![/]")

    except Exception as e:
        console.print(f"\n[bold red]Erro:[/] {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
@run_async
async def stats(ctx):
    """Mostra estatísticas do projeto"""
    try:
        config, db, analyzer, parser, editor, reporter = await initialize_system(
            ctx.obj['config_path']
        )

        metrics = await db.get_project_metrics()
        api_stats = await db.get_api_usage_stats()

        # Tabela de métricas
        table = Table(title="Estatísticas do Projeto")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", justify="right")

        table.add_row("Total de Arquivos", str(metrics['total_files_analyzed']))
        table.add_row("Total de Issues", str(sum(metrics['issues_by_priority'].values())))
        table.add_row("Mudanças Aplicadas", str(metrics['applied_changes']))
        table.add_row("Chamadas à API", str(api_stats['total_calls']))
        table.add_row("Custo Total API", f"${api_stats['total_cost']:.2f}")

        console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]Erro:[/] {str(e)}")
        sys.exit(1)


def main():
    """Função principal"""
    try:

        # Configura logging básico
        setup_basic_logging()

        # Verifica Python 3.9+
        if sys.version_info < (3, 9):
            console.print("[bold red]Erro: Python 3.9 ou superior é necessário[/]")
            sys.exit(1)

        # Executa CLI
        cli()

    except Exception as e:
        console.print(f"[bold red]Erro fatal:[/] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()