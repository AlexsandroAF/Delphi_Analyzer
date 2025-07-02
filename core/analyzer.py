"""
Analisador de código Delphi com suporte a IA.
Versão: 2.0
Autor: Sistema AIKA
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiofiles
import aiohttp
import chardet
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from core.database import CodeAnalysisDatabase, DatabaseError
from core.editor import DelphiCodeEditor
from core.models import (
    AnalysisResult,
    ApiInteraction,
    DelphiFile
)
from core.syntax_validator import DelphiSyntaxValidator


class AnalyzerError(Exception):
    """Exceção customizada para erros do analisador"""
    pass


class CircuitBreaker:
    """Implementa circuit breaker para chamadas à API"""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)

    async def can_execute(self) -> bool:
        async with self._lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                if (datetime.now() - self.last_failure_time).seconds >= self.reset_timeout:
                    self.logger.info("Circuit breaker mudando para half-open")
                    self.state = "half-open"
                    return True
                return False

            return True

    async def record_success(self):
        async with self._lock:
            self.failures = 0
            if self.state != "closed":
                self.logger.info("Circuit breaker fechando após sucesso")
            self.state = "closed"

    async def record_failure(self):
        async with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.now()
            if self.failures >= self.failure_threshold:
                if self.state != "open":
                    self.logger.warning("Circuit breaker abrindo após falhas consecutivas")
                self.state = "open"


class RateLimitController:
    """Controlador de rate limit com backoff exponencial"""

    def __init__(self, tokens_per_minute=50000, max_retries=3):
        self.tokens_per_minute = tokens_per_minute
        self.tokens_used = 0
        self.last_reset = datetime.now()
        self.max_retries = max_retries
        self.lock = asyncio.Lock()
        self._retry_delays = [2, 5, 10]
        self.logger = logging.getLogger(__name__)

    async def check_and_wait(self, tokens_needed: int) -> int:
        async with self.lock:
            now = datetime.now()
            if (now - self.last_reset).seconds >= 60:
                self.tokens_used = 0
                self.last_reset = now

            if self.tokens_used + tokens_needed > self.tokens_per_minute:
                wait_time = 60 - (now - self.last_reset).seconds
                self.logger.info(f"Rate limit atingido, aguardando {wait_time}s")
                return max(0, wait_time)

            self.tokens_used += tokens_needed
            return 0

    async def handle_rate_limit(self, retry_count: int) -> bool:
        if retry_count >= self.max_retries:
            self.logger.warning("Máximo de retentativas atingido")
            return False

        delay = self._retry_delays[min(retry_count, len(self._retry_delays) - 1)]
        self.logger.info(f"Aguardando {delay}s antes de nova tentativa")
        await asyncio.sleep(delay)
        return True


class DelphiAnalyzer:
    """Analisador principal de código Delphi com suporte a IA"""

    def __init__(self, project_dir: str, api_key: str, db: CodeAnalysisDatabase):
        """Inicializa o analisador"""
        self.project_dir = Path(project_dir)
        self.anthropic = Anthropic(api_key=api_key)
        self.db = db
        self.editor = DelphiCodeEditor()
        self.validator = DelphiSyntaxValidator()
        self.setup_logging()
        self.max_parallel_files = 4  # Valor padrão

        # Controles de API
        self.rate_limiter = RateLimitController()
        self.circuit_breaker = CircuitBreaker()

        # Pool de threads para I/O
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Session HTTP
        self.session = aiohttp.ClientSession()

        # Cache em memória
        self.memory_cache = {}

        # Task de limpeza
        self.cleanup_task = None
        self.logger.info(f"Analisador inicializado: {project_dir}")

    async def start(self):
        """Método para iniciar o analisador corretamente"""
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        await self.initialize()  # Novo método para inicialização assíncrona

    async def initialize(self):
        """Inicialização assíncrona"""
        try:
            # Inicializa session HTTP de forma assíncrona
            self.session = aiohttp.ClientSession()

            # Outras inicializações assíncronas necessárias
            self.logger.info("Analisador inicializado e pronto")
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            raise

    def setup_logging(self):
        """Configura sistema de logging"""
        import logging.handlers  # Importação explícita dos handlers

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Handler para arquivo com rotação
        rotating_handler = logging.handlers.RotatingFileHandler(
            '../analyzer.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        rotating_handler.setLevel(logging.DEBUG)

        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatação detalhada
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        rotating_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Remove handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Adiciona os novos handlers
        self.logger.addHandler(rotating_handler)
        self.logger.addHandler(console_handler)

    async def __aenter__(self):
        """Suporte para context manager"""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Cleanup ao sair do context"""
        await self.cleanup()

    async def cleanup(self):
        """Limpa recursos do analisador"""
        try:
            self.cleanup_task.cancel()
            await self.session.close()
            self.thread_pool.shutdown(wait=True)
            self.memory_cache.clear()
        except Exception as e:
            self.logger.error(f"Erro ao limpar recursos: {e}")

    async def analyze_project(self) -> Dict[str, Any]:
        """Analisa todo o projeto Delphi com processamento paralelo"""
        self.logger.info(f"Iniciando análise do projeto em {self.project_dir}")

        try:
            results = []
            stats = {
                'total_files': 0,
                'analyzed_files': 0,
                'fixed_files': 0,
                'errors': 0
            }

            # Lista todos os arquivos .pas
            delphi_files = list(Path(self.project_dir).rglob('*.pas'))

            if not delphi_files:
                self.logger.warning(f"Nenhum arquivo .pas encontrado em {self.project_dir}")
                raise ValueError(f"Nenhum arquivo Delphi (.pas) encontrado no diretório {self.project_dir}")

            self.logger.info(f"Encontrados {len(delphi_files)} arquivos .pas")
            stats['total_files'] = len(delphi_files)

            # Configura semáforo para limitar análises simultâneas
            semaphore = asyncio.Semaphore(self.max_parallel_files)

            async def analyze_with_semaphore(file_path: Path) -> Optional[AnalysisResult]:
                async with semaphore:
                    try:
                        self.logger.info(f"Analisando arquivo: {file_path}")
                        result = await self.analyze_file(file_path)
                        stats['analyzed_files'] += 1

                        if result and result.issues:
                            critical_fixes = [
                                issue for issue in result.issues
                                if issue.get('severity', 0) >= 4
                            ]
                            if critical_fixes:
                                success = await self.apply_suggestions(
                                    file_path,
                                    critical_fixes,
                                    min_priority=4
                                )
                                if success:
                                    stats['fixed_files'] += 1
                                    self.logger.info(f"Correções aplicadas em {file_path}")
                        return result
                    except Exception as e:
                        self.logger.error(f"Erro ao analisar {file_path}: {str(e)}")
                        stats['errors'] += 1
                        return None

            # Executa análises em paralelo
            tasks = [analyze_with_semaphore(file_path) for file_path in delphi_files]
            results = await asyncio.gather(*tasks)
            results = [r for r in results if r]  # Remove resultados None

            # Gera relatório final
            report_path = Path(self.project_dir) / 'analysis_report.md'
            async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
                report_content = await self._generate_project_report(results, stats)
                await f.write(report_content)

            self.logger.info("Análise do projeto concluída")
            self.logger.info(f"Relatório gerado em: {report_path}")

            return {
                'statistics': stats,
                'results': results,
                'report_path': str(report_path)
            }

        except Exception as e:
            self.logger.error(f"Erro na análise do projeto: {str(e)}")
            raise

    async def _read_delphi_file_safe(self, file_path: Path) -> DelphiFile:
        """Lê arquivo Delphi com proteções e detecção de encoding"""
        try:
            # Verifica se arquivo existe
            if not file_path.exists():
                raise AnalyzerError(f"Arquivo não encontrado: {file_path}")

            # Lê conteúdo com detecção segura de encoding
            async with aiofiles.open(file_path, 'rb') as f:
                raw_data = await f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'

            async with aiofiles.open(file_path, encoding=encoding) as f:
                content = await f.read()

            # Extrai nome da unit
            unit_match = self.validator.patterns['unit_decl'].search(content)
            unit_name = unit_match.group(1) if unit_match else file_path.stem

            # Limpa caracteres inválidos
            content = content.replace('\x00', '')

            # Calcula hash do conteúdo
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # Obtém data de modificação
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

            # Cria e retorna objeto DelphiFile
            delphi_file = DelphiFile(
                path=file_path,
                content=content,
                encoding=encoding,
                unit_name=unit_name,
                content_hash=content_hash,
                last_modified=last_modified
            )

            self.logger.debug(f"Arquivo lido com sucesso: {file_path} (encoding: {encoding})")
            return delphi_file

        except UnicodeDecodeError as e:
            self.logger.error(f"Erro de encoding ao ler {file_path}: {str(e)}")
            raise AnalyzerError(f"Erro de encoding ao ler arquivo: {str(e)}")
        except Exception as e:
            self.logger.error(f"Erro ao ler arquivo {file_path}: {str(e)}")
            raise AnalyzerError(f"Erro ao ler arquivo: {str(e)}")

    async def _generate_project_report(self, results: List[AnalysisResult], stats: Dict[str, int]) -> str:
        """Gera relatório detalhado da análise do projeto"""
        sections = []

        # Cabeçalho
        sections.append("# Relatório de Análise de Código Delphi")
        sections.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append("\n## Estatísticas Gerais")
        sections.append(f"- Total de arquivos: {stats['total_files']}")
        sections.append(f"- Arquivos analisados: {stats['analyzed_files']}")
        sections.append(f"- Arquivos corrigidos: {stats['fixed_files']}")
        sections.append(f"- Erros encontrados: {stats['errors']}")

        # Problemas por prioridade
        priority_issues = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for result in results:
            for issue in result.issues:
                priority = issue.get('severity', 1)
                priority_issues[priority] += 1

        sections.append("\n## Distribuição de Problemas")
        for priority, count in priority_issues.items():
            sections.append(f"- Prioridade {priority}: {count} problemas")

        # Detalhes por arquivo
        sections.append("\n## Detalhes por Arquivo")
        for result in results:
            if result.issues:
                sections.append(f"\n### {result.file_path}")
                sections.append(f"Prioridade geral: {result.priority}")
                sections.append("\nProblemas encontrados:")
                for issue in result.issues:
                    sections.append(f"\n- **{issue['type']}** (Severidade: {issue['severity']})")
                    sections.append(f"  - {issue['description']}")
                    sections.append(f"  - Localização: {issue['location']}")
                    if 'suggested_fix' in issue:
                        sections.append("  - Sugestão de correção:")
                        sections.append(f"```delphi\n{issue['suggested_fix']}\n```")

        # Métricas
        sections.append("\n## Métricas do Projeto")
        total_complexity = sum(r.metrics.get('complexidade_ciclomatica', 0) for r in results)
        total_loc = sum(r.metrics.get('linhas_codigo', 0) for r in results)
        sections.append(f"- Complexidade ciclomática total: {total_complexity}")
        sections.append(f"- Total de linhas de código: {total_loc}")

        # Recomendações
        sections.append("\n## Recomendações")
        if priority_issues[5] > 0:
            sections.append("- **URGENTE**: Corrija imediatamente os problemas de prioridade 5")
        if priority_issues[4] > 0:
            sections.append("- **IMPORTANTE**: Planeje a correção dos problemas de prioridade 4")
        if sum(priority_issues[i] for i in [1, 2, 3]) > 0:
            sections.append("- Considere revisar os problemas de menor prioridade em futuras sprints")

        return '\n'.join(sections)

    async def _periodic_cleanup(self):
        """Executa limpeza periódica"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hora
                await self._cleanup_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erro na limpeza periódica: {e}")

    async def _cleanup_resources(self):
        """Limpa recursos do sistema"""
        try:
            self.memory_cache.clear()
            await self.db.cleanup_old_cache()
            await self.session.close()
            self.session = aiohttp.ClientSession()
        except Exception as e:
            self.logger.error(f"Erro ao limpar recursos: {e}")

    async def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Analisa um arquivo Delphi"""
        self.logger.info(f"Iniciando análise: {file_path}")
        start_time = time.time()

        try:
            # Verifica se arquivo existe
            if not file_path.exists():
                raise AnalyzerError(f"Arquivo não encontrado: {file_path}")

            # Lê e prepara arquivo
            delphi_file = await self._read_delphi_file_safe(file_path)
            self.current_unit_name = delphi_file.unit_name

            # Verifica cache
            cache_result = await self._check_cache(file_path, delphi_file.content_hash)
            if cache_result:
                self.logger.info(f"Usando cache para {file_path}")
                return cache_result

            # Analisa código
            self.logger.info(f"Analisando conteúdo de {file_path}")
            result = await self._analyze_content(delphi_file)

            if not result:
                raise AnalyzerError(f"Falha ao analisar {file_path}")

            result.execution_time = time.time() - start_time

            # Salva no cache
            await self._save_to_cache(str(file_path), delphi_file.content_hash, result)

            self.logger.info(f"Análise concluída: {file_path}")
            return result

        except Exception as e:
            self.logger.error(f"Erro ao analisar {file_path}: {str(e)}")
            raise AnalyzerError(f"Erro ao analisar arquivo: {str(e)}")

    async def _check_cache(self, file_path: Path, content_hash: str) -> Optional[AnalysisResult]:
        """Verifica cache em memória e banco"""
        cache_key = f"{file_path}:{content_hash}"

        # Verifica cache em memória
        if cache_key in self.memory_cache:
            self.logger.info(f"Cache em memória encontrado: {file_path}")
            return self.memory_cache[cache_key]

        # Verifica cache no banco
        cache_result = await self.db.get_cached_analysis(str(file_path), content_hash)
        if cache_result:
            self.logger.info(f"Cache em banco encontrado: {file_path}")
            result = AnalysisResult(**cache_result)
            self.memory_cache[cache_key] = result
            return result

        return None

    async def _save_to_cache(self, file_path: str, content_hash: str, result: AnalysisResult) -> bool:
        """Salva resultado no cache"""
        try:
            cache_key = f"{file_path}:{content_hash}"
            self.memory_cache[cache_key] = result
            await self.db.save_to_cache(file_path, content_hash, asdict(result))
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar cache: {str(e)}")
            return False

    def _combine_section_results(self, section_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combina resultados de diferentes seções em um único resultado"""
        combined = {
            'issues': [],
            'metrics': {
                'complexidade_ciclomatica': 0,
                'linhas_codigo': 0,
                'pontos_criticos': 0,
                'cobertura_try_except': 0
            },
            'suggestions': [],
            'priority': 1
        }

        if not section_results:
            return combined

        try:
            # Combina issues
            for result in section_results:
                if 'issues' in result:
                    combined['issues'].extend(result['issues'])

            # Combina métricas
            for result in section_results:
                if 'metrics' in result:
                    for metric, value in result['metrics'].items():
                        if isinstance(value, (int, float)):
                            combined['metrics'][metric] = combined['metrics'].get(metric, 0) + value

            # Combina sugestões
            for result in section_results:
                if 'suggestions' in result:
                    combined['suggestions'].extend(result['suggestions'])

            # Calcula prioridade máxima
            priorities = [
                result.get('priority', 1)
                for result in section_results
                if 'priority' in result
            ]
            combined['priority'] = max(priorities) if priorities else 1

            # Remove duplicatas
            combined['issues'] = list({
                                          json.dumps(issue, sort_keys=True): issue
                                          for issue in combined['issues']
                                      }.values())

            combined['suggestions'] = list({
                                               json.dumps(sugg, sort_keys=True): sugg
                                               for sugg in combined['suggestions']
                                           }.values())

            return combined

        except Exception as e:
            self.logger.error(f"Erro ao combinar resultados: {str(e)}")
            return combined

    async def _analyze_content(self, delphi_file: DelphiFile) -> AnalysisResult:
        """Analisa conteúdo do arquivo"""
        # Divide código em regiões/seções
        regions = await self._split_by_regions_safe(delphi_file.content)
        if not regions:
            self.logger.warning(f"Sem regiões em {delphi_file.path}, usando seções")
            regions = await self._split_into_sections_safe(delphi_file)

        # Analisa cada região em paralelo
        tasks = [self._analyze_code_section(region) for region in regions]
        section_results = await asyncio.gather(*tasks)
        section_results = [r for r in section_results if r]

        # Combina resultados
        combined = self._combine_section_results(section_results)

        return AnalysisResult(
            file_path=str(delphi_file.path),
            timestamp=datetime.now(),
            issues=combined['issues'],
            metrics=combined['metrics'],
            suggestions=combined['suggestions'],
            priority=combined['priority'],
            content_hash=delphi_file.content_hash,
            execution_time=0  # Será definido depois
        )

    async def _analyze_large_file(self, file_path: Path) -> AnalysisResult:
        """Analisa arquivo grande dividindo em partes"""
        MAX_SECTION_SIZE = 100_000  # 100KB por seção

        try:
            delphi_file = await self._read_delphi_file_safe(file_path)
            sections = await self._split_large_file(delphi_file, MAX_SECTION_SIZE)

            all_results = []
            for section in sections:
                result = await self._analyze_code_section(section)
                if result:
                    all_results.append(result)

            combined = self._combine_section_results(all_results)

            return AnalysisResult(
                file_path=str(file_path),
                timestamp=datetime.now(),
                issues=combined['issues'],
                metrics=combined['metrics'],
                suggestions=combined['suggestions'],
                priority=combined['priority'],
                content_hash=delphi_file.content_hash,
                execution_time=0
            )

        except Exception as e:
            self.logger.error(f"Erro ao analisar arquivo grande {file_path}: {str(e)}")
            raise

    async def _split_large_file(self, delphi_file: DelphiFile, max_size: int) -> List[Dict[str, Any]]:
        """Divide arquivo grande em seções menores"""
        sections = []
        content = delphi_file.content
        current_pos = 0

        while current_pos < len(content):
            # Encontra próximo ponto de divisão seguro
            end_pos = self._find_safe_split_point(content, current_pos, max_size)

            section_content = content[current_pos:end_pos]
            sections.append({
                'type': 'partial',
                'content': section_content,
                'unit_name': delphi_file.unit_name
            })

            current_pos = end_pos

        return sections

    def _find_safe_split_point(self, content: str, start: int, max_size: int) -> int:
        """Encontra ponto seguro para dividir o código"""
        end = start + max_size
        if end >= len(content):
            return len(content)

#PARTE 3
        # Procura fim de procedimento/função
        possible_points = [
            pos for pos in [
                content.find('end;', end - 100, end + 100),
                content.find('begin', end - 100, end + 100),
                content.rfind('.', end - 100, end)
            ] if pos >= start and pos <= end + 100
        ]

        if possible_points:
            return max(possible_points) + 4  # +4 para incluir 'end;'

        # Se não encontrar ponto ideal, usa quebra de linha
        line_break = content.find('\n', end - 50, end + 50)
        if line_break > start:
            return line_break

        return end

    async def _split_by_regions_safe(self, content: str) -> List[Dict[str, Any]]:
        """Versão segura do split por regiões"""
        try:
            return self._split_by_regions(content)
        except Exception as e:
            self.logger.error(f"Erro ao dividir por regiões: {str(e)}")
            return []

    async def _split_into_sections_safe(self, delphi_file: DelphiFile) -> List[Dict[str, Any]]:
        """Versão segura do split em seções"""
        try:
            return self._split_into_sections(delphi_file)
        except Exception as e:
            self.logger.error(f"Erro ao dividir em seções: {str(e)}")
            return []

    def _split_by_regions(self, content: str) -> List[Dict[str, Any]]:
        """Divide o código por regiões {$REGION}"""
        regions = []
        lines = content.split('\n')
        current_region = None
        current_content = []

        for line in lines:
            stripped = line.strip().lower()

            # Detecta início de região
            if '{$region' in stripped:
                # Fecha região atual se existir
                if current_region:
                    regions.append({
                        'type': current_region,
                        'content': '\n'.join(current_content),
                        'unit_name': self.current_unit_name
                    })

                # Extrai nome da nova região
                region_name = stripped[stripped.find('{$region'):].split('}')[0]
                region_name = region_name.replace('{$region', '').replace("'", "").replace('"', '').strip()
                current_region = region_name
                current_content = []

            # Detecta fim de região
            elif '{$endregion}' in stripped.lower():
                if current_region:
                    regions.append({
                        'type': current_region,
                        'content': '\n'.join(current_content),
                        'unit_name': self.current_unit_name
                    })
                current_region = None
                current_content = []

            # Adiciona linha à região atual
            elif current_region:
                current_content.append(line)

        # Fecha última região se existir
        if current_region and current_content:
            regions.append({
                'type': current_region,
                'content': '\n'.join(current_content),
                'unit_name': self.current_unit_name
            })

        return regions

    def _split_into_sections(self, delphi_file: DelphiFile) -> List[Dict[str, Any]]:
        """Divide arquivo em seções lógicas"""
        sections = []
        content = delphi_file.content

        # Interface
        interface_match = re.search(
            r'interface(.*?)implementation',
            content,
            re.DOTALL
        )
        if interface_match:
            sections.append({
                'type': 'interface',
                'content': interface_match.group(1),
                'unit_name': delphi_file.unit_name
            })

        # Implementation
        impl_match = re.search(
            r'implementation(.*?)(?:initialization|finalization|end\.)',
            content,
            re.DOTALL
        )
        if impl_match:
            sections.append({
                'type': 'implementation',
                'content': impl_match.group(1),
                'unit_name': delphi_file.unit_name
            })

        # Initialization
        init_match = re.search(
            r'initialization(.*?)(?:finalization|end\.)',
            content,
            re.DOTALL
        )
        if init_match:
            sections.append({
                'type': 'initialization',
                'content': init_match.group(1),
                'unit_name': delphi_file.unit_name
            })

        # Finalization
        final_match = re.search(
            r'finalization(.*?)end\.',
            content,
            re.DOTALL
        )
        if final_match:
            sections.append({
                'type': 'finalization',
                'content': final_match.group(1),
                'unit_name': delphi_file.unit_name
            })

        return sections

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _analyze_code_section(self, section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analisa uma seção de código usando IA"""
        try:
            # Otimiza código para análise
            optimized_code = self._optimize_code_for_analysis(section['content'])

            # Cria prompt
            prompt = self._create_analysis_prompt(
                section['type'],
                section['unit_name'],
                optimized_code
            )

            # Chama API com circuit breaker
            response = await self._call_claude_api(prompt)

            # Valida resposta
            validated_response = self._validate_analysis_result(response)

            # Registra interação
            if validated_response:
                await self._register_api_interaction(prompt, validated_response)

            return validated_response

        except Exception as e:
            self.logger.error(f"Erro ao analisar seção: {str(e)}")
            return self._validate_analysis_result(None)

    async def _register_api_interaction(self, prompt: str, response: Dict[str, Any]) -> None:
        """Registra interação com a API no banco de dados"""
        try:
            interaction = ApiInteraction(
                timestamp=datetime.now(),
                prompt=prompt,
                response=json.dumps(response),
                tokens_used=len(prompt) // 4,
                cost=0.0,  # Atualize conforme necessário
                success=True,
                latency=0,
                model="claude-3-haiku-20240307"
            )

            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    await self.db.save_api_interaction(interaction)
                    break
                except DatabaseError as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    self.logger.warning(f"Tentativa {retry_count} de salvar interação falhou, tentando novamente...")
                    await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Erro ao registrar interação: {str(e)}")

    def _optimize_code_for_analysis(self, content: str) -> str:
        """Otimiza o código para análise"""
        try:
            lines = content.split('\n')
            optimized_lines = []
            in_comment_block = False
            in_string = False
            string_char = None
            indent_stack = []

            for line in lines:
                # Preserva indentação
                original_indent = len(line) - len(line.lstrip())
                line = line.strip()

                if not line:
                    continue

                # Gerencia blocos de comentários
                if not in_string:
                    if line.startswith('(*') or line.startswith('{'):
                        in_comment_block = True
                        continue
                    if line.endswith('*)') or line.endswith('}'):
                        in_comment_block = False
                        continue
                    if in_comment_block:
                        continue

                # Processa caracteres e strings
                i = 0
                processed_line = ""
                while i < len(line):
                    char = line[i]

                    if char in ['"', "'"]:
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                            string_char = None

                    if not in_string:
                        # Remove comentários inline
                        if char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                            break
                        if char == '{':
                            while i < len(line) and line[i] != '}':
                                i += 1
                            i += 1
                            continue

                    processed_line += char
                    i += 1

                line = processed_line.strip()
                if not line:
                    continue

                # Simplifica strings longas
                if not in_string:
                    line = re.sub(r"'[^']{50,}'", "'...'", line)
                    line = re.sub(r'"[^"]{50,}"', '"..."', line)

                # Remove múltiplos espaços
                line = re.sub(r'\s+', ' ', line)

                # Gerencia estrutura de blocos
                if 'begin' in line.lower():
                    indent_stack.append(original_indent)
                elif 'end' in line.lower() and indent_stack:
                    original_indent = indent_stack.pop()

                # Restaura indentação
                line = ' ' * original_indent + line
                optimized_lines.append(line)

            # Remove múltiplas linhas em branco
            optimized_content = '\n'.join(optimized_lines)
            optimized_content = re.sub(r'\n\s*\n', '\n', optimized_content)

            return optimized_content

        except Exception as e:
            self.logger.error(f"Erro ao otimizar código: {str(e)}")
            return content

    def _create_analysis_prompt(self, section_type: str, unit_name: str, optimized_code: str) -> str:
        """Cria prompt otimizado para análise"""
        return (
            f"Analise este código Delphi da unit {unit_name}, "
            f"seção {section_type}. Este código foi otimizado para análise, removendo comentários e "
            "simplificando strings longas, mas mantendo toda a estrutura lógica.\n\n"
            "Foque em:\n"
            "1. Segurança (SQL Injection, variáveis não inicializadas, buffer overflows)\n"
            "2. Performance (loops ineficientes, operações de IO redundantes)\n"
            "3. Robustez (tratamento de exceções, validações)\n"
            "4. Boas práticas (nomes de variáveis, estrutura do código)\n"
            "5. Bugs potenciais (vazamentos de memória, race conditions)\n\n"
            f"Código Original:\n{optimized_code}\n\n"
            "Forneça análise em JSON:\n"
            "{\n"
            '    "issues": [\n'
            "        {\n"
            '            "type": "security|performance|robustness|bug",\n'
            '            "description": "Descrição detalhada do problema",\n'
            '            "location": "Trecho exato do código com problema",\n'
            '            "severity": 1-5,\n'
            '            "corrected_code": "Código corrigido que substituirá o trecho em location",\n'
            '            "explanation": "Explicação técnica detalhada da correção"\n'
            "        }\n"
            "    ],\n"
            '    "metrics": {\n'
            '        "complexidade_ciclomatica": 0,\n'
            '        "linhas_codigo": 0,\n'
            '        "pontos_criticos": 0,\n'
            '        "cobertura_try_except": 0\n'
            "    },\n"
            '    "suggestions": [\n'
            "        {\n"
            '            "type": "improvement|optimization|security",\n'
            '            "description": "Sugestão de melhoria",\n'
            '            "priority": 1-5,\n'
            '            "implementation": "Como implementar"\n'
            "        }\n"
            "    ],\n"
            '    "priority": 1-5,\n'
            '    "full_corrected_code": "Código completo da seção já com todas as correções aplicadas"\n'
            "}\n\n"
            "IMPORTANTE:\n"
            "- Em 'location' forneça o trecho EXATO do código que precisa ser substituído\n"
            "- Em 'corrected_code' forneça o código novo que substituirá o trecho em 'location'\n"
            "- 'full_corrected_code' deve conter a seção inteira já corrigida\n"
            "- Use apenas valores numéricos sem símbolos especiais (como %) nas métricas"
        )

    async def _call_claude_api(self, prompt: str) -> Dict[str, Any]:
        """Chama a API do Claude com circuit breaker e retries"""
        if not await self.circuit_breaker.can_execute():
            raise AnalyzerError("Circuit breaker aberto")

        try:
            wait_time = await self.rate_limiter.check_and_wait(len(prompt) // 4)
            if wait_time > 0:
                self.logger.info(f"Aguardando {wait_time}s pelo rate limit...")
                await asyncio.sleep(wait_time)

            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            if response and response.content:
                await self.circuit_breaker.record_success()
                content = response.content[0].text

                try:
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        json_str = json_match.group(0)
                        json_str = self._preprocess_json(json_str)
                        return json.loads(json_str)

                    raise ValueError("JSON não encontrado na resposta")

                except json.JSONDecodeError as je:
                    self.logger.error(f"Erro ao decodificar JSON: {str(je)}")
                    raise

            raise ValueError("Resposta vazia da API")

        except Exception as e:
            await self.circuit_breaker.record_failure()
            self.logger.error(f"Erro na chamada à API: {str(e)}")
            raise

    def _validate_analysis_result(self, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Valida e normaliza resultado da análise"""
        template = {
            "issues": [],
            "metrics": {
                "complexidade_ciclomatica": 0,
                "linhas_codigo": 0,
                "pontos_criticos": 0,
                "cobertura_try_except": 0
            },
            "suggestions": [],
            "priority": 1
        }

        if not result:
            return template

        validated = template.copy()

        try:
            # Valida e copia issues
            if "issues" in result and isinstance(result["issues"], list):
                validated_issues = []
                for issue in result["issues"]:
                    if self._validate_issue(issue):
                        validated_issues.append(issue)
                validated["issues"] = validated_issues

            # Valida e copia métricas
            if "metrics" in result and isinstance(result["metrics"], dict):
                for key in template["metrics"]:
                    if key in result["metrics"]:
                        try:
                            validated["metrics"][key] = int(float(result["metrics"][key]))
                        except (ValueError, TypeError):
                            validated["metrics"][key] = 0

            # Valida e copia sugestões
            if "suggestions" in result and isinstance(result["suggestions"], list):
                validated_suggestions = []
                for suggestion in result["suggestions"]:
                    if self._validate_suggestion(suggestion):
                        validated_suggestions.append(suggestion)
                validated["suggestions"] = validated_suggestions

            # Valida prioridade
            if "priority" in result and isinstance(result["priority"], (int, float)):
                validated["priority"] = min(max(int(result["priority"]), 1), 5)

            return validated

        except Exception as e:
            self.logger.error(f"Erro ao validar resultado: {str(e)}")
            return template

    def _validate_issue(self, issue: Dict[str, Any]) -> bool:
        """Valida uma issue individual"""
        required_fields = {
            "type": str,
            "description": str,
            "location": str,
            "severity": (int, float)
        }

        try:
            for field, field_type in required_fields.items():
                if field not in issue or not isinstance(issue[field], field_type):
                    return False

            # Normaliza severity para inteiro entre 1-5
            issue["severity"] = min(max(int(float(issue["severity"])), 1), 5)

            # Valida tipo de issue
            valid_types = {"security", "performance", "robustness", "bug"}
            if issue["type"].lower() not in valid_types:
                return False

            return True

        except Exception:
            return False

    def _validate_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Valida uma sugestão individual"""
        required_fields = {
            "type": str,
            "description": str,
            "priority": (int, float),
            "implementation": str
        }

        try:
            for field, field_type in required_fields.items():
                if field not in suggestion or not isinstance(suggestion[field], field_type):
                    return False

            # Normaliza prioridade
            suggestion["priority"] = min(max(int(float(suggestion["priority"])), 1), 5)

            # Valida tipo de sugestão
            valid_types = {"improvement", "optimization", "security"}
            if suggestion["type"].lower() not in valid_types:
                return False

            return True

        except Exception:
            return False

    def _preprocess_json(self, json_str: str) -> str:
        """Pré-processa o JSON para corrigir problemas comuns"""
        try:
            # Remove caracteres especiais das métricas
            json_str = re.sub(r':\s*(\d+)%', r': \1', json_str)

            # Remove caracteres inválidos
            json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)

            # Corrige vírgulas trailing
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # Corrige valores booleanos
            json_str = re.sub(r':\s*true\b', ': true', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r':\s*false\b', ': false', json_str, flags=re.IGNORECASE)

            # Corrige strings não escapadas
            def escape_quotes(match):
                return '"' + match.group(1).replace('"', '\\"') + '"'

            json_str = re.sub(r'(?<!\\)"(.*?)(?<!\\)"', escape_quotes, json_str)

            return json_str

        except Exception as e:
            self.logger.error(f"Erro no pré-processamento do JSON: {str(e)}")
            return json_str

    async def export_metrics(self, output_path: Path):
        """Exporta métricas detalhadas do projeto"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'project_metrics': await self.db.get_project_metrics(),
                'files': {}
            }

            for file_path in self.project_dir.rglob('*.pas'):
                file_metrics = await self.db.get_file_metrics(str(file_path))
                if file_metrics:
                    metrics['files'][str(file_path)] = file_metrics

            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metrics, indent=2))

            self.logger.info(f"Métricas exportadas para {output_path}")

        except Exception as e:
            self.logger.error(f"Erro ao exportar métricas: {str(e)}")
            raise AnalyzerError(f"Erro ao exportar métricas: {str(e)}")

    async def apply_suggestions(self, file_path: Path, analysis_result: Dict, min_severity: int = 4, backup_path: Optional[str] = None) -> bool:
        """Aplica correções ao código"""
        try:
            # Filtra issues pela severidade mínima
            issues = [
                issue for issue in analysis_result.get('issues', [])
                if issue.get('severity', 0) >= min_severity
            ]

            if not issues:
                self.logger.info(f"Sem correções para aplicar em {file_path}")
                return False

            # Lê arquivo atual
            delphi_file = await self._read_delphi_file_safe(file_path)
            content = delphi_file.content

            # Prepara mudanças
            changes = []
            for issue in issues:
                old_code = issue.get('location')
                new_code = issue.get('corrected_code')

                if old_code and new_code and old_code in content:
                    changes.append({
                        'type': 'fix',
                        'description': issue['description'],
                        'old_code': old_code,
                        'new_code': new_code,
                        'priority': issue.get('severity', 1)
                    })

            # Aplica mudanças
            if changes:
                success, backup_path = await self.editor.apply_changes(
                    file_path,
                    changes,
                    create_backup=True
                )

                if success:
                    self.logger.info(f"Correções aplicadas em {file_path}. Backup: {backup_path}")
                    # Registra mudanças no banco para histórico
                    await self.db.save_changes(
                        file_path=str(file_path),
                        changes=changes,
                        backup_path=backup_path
                    )

                return success

            return False

        except Exception as e:
            self.logger.error(f"Erro ao aplicar correções em {file_path}: {str(e)}")
            return False


    def __del__(self):
        """Destructor para garantir limpeza de recursos"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            self.logger.error(f"Erro no cleanup: {str(e)}")