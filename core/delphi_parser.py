import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Set

import networkx as nx

from core.models import CodeBlock


@dataclass
class DelphiUnit:
    """Representa uma unit Delphi completa após parsing"""
    name: str
    file_path: Path
    uses_interface: List[str] = field(default_factory=list)
    uses_implementation: List[str] = field(default_factory=list)
    declarations: List[CodeBlock] = field(default_factory=list)
    implementations: List[CodeBlock] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    exports: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)


class DelphiParser:
    """Parser avançado de código Delphi"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_patterns()

    def setup_patterns(self):
        """Configura padrões regex para parsing"""
        self.patterns = {
            'unit_decl': re.compile(r'unit\s+([a-zA-Z][\w\.]*)\s*;'),
            'uses_clause': re.compile(r'uses\s+([^;]+);'),
            'procedure_decl': re.compile(
                r'procedure\s+([a-zA-Z]\w*)'
                r'(?:\s*\(\s*([^)]*)\s*\))?'
                r'(?:\s*:\s*([^;]+))?'
                r'\s*;'
            ),
            'function_decl': re.compile(
                r'function\s+([a-zA-Z]\w*)'
                r'(?:\s*\(\s*([^)]*)\s*\))?'
                r'\s*:\s*([^;]+)'
                r'\s*;'
            ),
            'class_decl': re.compile(
                r'(\w+)\s*=\s*class\s*'
                r'(?:\((.*?)\))?'
                r'(?:\s+(sealed|abstract))?'
            ),
            'var_decl': re.compile(r'var\s+([^;]+);'),
            'const_decl': re.compile(r'const\s+([^;]+);'),
            'type_decl': re.compile(r'type\s+([^;]+);'),
            'property_decl': re.compile(
                r'property\s+(\w+)\s*:\s*(\w+)'
                r'(?:\s+index\s+\d+)?'
                r'(?:\s+read\s+\w+)?'
                r'(?:\s+write\s+\w+)?'
                r'\s*;'
            ),
            'method_impl': re.compile(
                r'(procedure|function)\s+'
                r'(\w+)\.(\w+)'
                r'(?:\s*\(\s*([^)]*)\s*\))?'
                r'(?:\s*:\s*([^;]+))?'
                r'\s*;'
            )
        }

    async def parse_file(self, file_path: Path) -> DelphiUnit:
        """Realiza parsing completo de um arquivo Delphi"""
        try:
            # Lê arquivo
            content = await self._read_file(file_path)

            # Extrai nome da unit
            unit_match = self.patterns['unit_decl'].search(content)
            if not unit_match:
                raise ValueError(f"Nome da unit não encontrado em {file_path}")

            unit_name = unit_match.group(1)

            # Cria unidade
            unit = DelphiUnit(
                name=unit_name,
                file_path=file_path
            )

            # Processa seções principais
            self._parse_interface_section(content, unit)
            self._parse_implementation_section(content, unit)

            # Analisa dependências
            self._analyze_dependencies(unit)

            return unit

        except Exception as e:
            self.logger.error(f"Erro ao fazer parsing de {file_path}: {str(e)}")
            raise

    def _parse_interface_section(self, content: str, unit: DelphiUnit):
        """Processa seção interface da unit"""
        interface_match = re.search(
            r'interface(.*?)implementation',
            content,
            re.DOTALL
        )

        if not interface_match:
            return

        interface_content = interface_match.group(1)

        # Processa uses
        uses_matches = self.patterns['uses_clause'].finditer(interface_content)
        for match in uses_matches:
            units = [u.strip() for u in match.group(1).split(',')]
            unit.uses_interface.extend(units)
            unit.dependencies.update(units)

        # Processa declarações
        declarations = []

        # Processa classes
        class_matches = self.patterns['class_decl'].finditer(interface_content)
        for match in class_matches:
            class_name = match.group(1)
            parent_class = match.group(2)
            modifiers = match.group(3)

            class_block = CodeBlock(
                type='class',
                name=class_name,
                content=match.group(0),
                start_line=self._get_line_number(interface_content, match.start()),
                end_line=self._get_line_number(interface_content, match.end())
            )

            if parent_class:
                unit.dependencies.add(parent_class.strip())

            declarations.append(class_block)
            unit.exports.add(class_name)

        # Processa procedures e functions
        proc_matches = self.patterns['procedure_decl'].finditer(interface_content)
        func_matches = self.patterns['function_decl'].finditer(interface_content)

        for match in proc_matches:
            proc_name = match.group(1)
            params = match.group(2)

            proc_block = CodeBlock(
                type='procedure',
                name=proc_name,
                content=match.group(0),
                start_line=self._get_line_number(interface_content, match.start()),
                end_line=self._get_line_number(interface_content, match.end())
            )

            if params:
                self._process_parameters(params, unit)

            declarations.append(proc_block)
            unit.exports.add(proc_name)

        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            func_block = CodeBlock(
                type='function',
                name=func_name,
                content=match.group(0),
                start_line=self._get_line_number(interface_content, match.start()),
                end_line=self._get_line_number(interface_content, match.end())
            )

            if params:
                self._process_parameters(params, unit)
            if return_type:
                unit.dependencies.add(return_type.strip())

            declarations.append(func_block)
            unit.exports.add(func_name)

        unit.declarations = declarations

    def _parse_implementation_section(self, content: str, unit: DelphiUnit):
        """Processa seção implementation da unit"""
        impl_match = re.search(
            r'implementation(.*?)(?:initialization|finalization|end\.)',
            content,
            re.DOTALL
        )

        if not impl_match:
            return

        impl_content = impl_match.group(1)

        # Processa uses da implementation
        uses_matches = self.patterns['uses_clause'].finditer(impl_content)
        for match in uses_matches:
            units = [u.strip() for u in match.group(1).split(',')]
            unit.uses_implementation.extend(units)
            unit.dependencies.update(units)

        # Processa implementações de métodos
        implementations = []
        method_matches = self.patterns['method_impl'].finditer(impl_content)

        for match in method_matches:
            method_type = match.group(1)
            class_name = match.group(2)
            method_name = match.group(3)
            params = match.group(4)
            return_type = match.group(5)

            method_block = CodeBlock(
                type=method_type.lower(),
                name=f"{class_name}.{method_name}",
                content=match.group(0),
                start_line=self._get_line_number(impl_content, match.start()),
                end_line=self._get_line_number(impl_content, match.end())
            )

            if params:
                self._process_parameters(params, unit)
            if return_type:
                unit.dependencies.add(return_type.strip())

            implementations.append(method_block)

        unit.implementations = implementations

    def _process_parameters(self, params: str, unit: DelphiUnit):
        """Processa parâmetros de métodos para extrair dependências"""
        param_list = params.split(';')
        for param in param_list:
            param = param.strip()
            if ':' in param:
                param_type = param.split(':')[1].strip()
                if param_type:
                    unit.dependencies.add(param_type)

    def _analyze_dependencies(self, unit: DelphiUnit):
        """Analisa e organiza dependências da unit"""
        # Remove auto-referências
        unit.dependencies.discard(unit.name)

        # Separa imports (dependências diretas) de exports (interfaces expostas)
        unit.imports = unit.dependencies.copy()

        # Adiciona tipos exportados das declarações
        for decl in unit.declarations:
            if decl.type in ['class', 'type']:
                unit.exports.add(decl.name)

    def _get_line_number(self, content: str, pos: int) -> int:
        """Obtém número da linha baseado na posição no texto"""
        return content.count('\n', 0, pos) + 1

    async def _read_file(self, file_path: Path) -> str:
        """Lê arquivo com detecção de encoding"""
        try:
            import chardet
            import aiofiles

            async with aiofiles.open(file_path, 'rb') as f:
                raw_data = await f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'

            async with aiofiles.open(file_path, encoding=encoding) as f:
                return await f.read()

        except Exception as e:
            self.logger.error(f"Erro ao ler arquivo {file_path}: {str(e)}")
            raise

    async def analyze_dependencies(self, project_dir: Path) -> nx.DiGraph:
        """Analisa dependências entre units do projeto"""
        try:
            # Cria grafo direcionado
            dep_graph = nx.DiGraph()

            # Processa todas as units
            for file_path in project_dir.rglob('*.pas'):
                unit = await self.parse_file(file_path)

                # Adiciona nó para a unit
                dep_graph.add_node(unit.name, file_path=str(file_path))

                # Adiciona dependências
                for dep in unit.dependencies:
                    dep_graph.add_edge(unit.name, dep)

            return dep_graph

        except Exception as e:
            self.logger.error(f"Erro ao analisar dependências: {str(e)}")
            raise

    def find_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]:
        """Encontra dependências circulares no projeto"""
        try:
            return list(nx.simple_cycles(graph))
        except Exception as e:
            self.logger.error(f"Erro ao procurar dependências circulares: {str(e)}")
            return []

    def get_dependency_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calcula métricas de dependências"""
        try:
            metrics = {
                'total_units': len(graph.nodes),
                'total_dependencies': len(graph.edges),
                'avg_dependencies': len(graph.edges) / len(graph.nodes) if graph.nodes else 0,
                'max_dependencies': max(d for _, d in graph.out_degree()) if graph.nodes else 0,
                'circular_dependencies': len(self.find_circular_dependencies(graph)),
                'isolated_units': len(list(nx.isolates(graph)))
            }

            # Centralidade
            metrics['most_dependent'] = max(
                graph.nodes,
                key=lambda n: graph.out_degree(n),
                default=None
            )

            metrics['most_depended_on'] = max(
                graph.nodes,
                key=lambda n: graph.in_degree(n),
                default=None
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Erro ao calcular métricas: {str(e)}")
            return {}