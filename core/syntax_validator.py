import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import chardet


@dataclass
class SyntaxError:
    line: int
    column: int
    message: str
    severity: int  # 1-5
    error_type: str
    suggestion: Optional[str] = None


class DelphiSyntaxValidator:
    """Validador avançado de sintaxe Delphi"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_patterns()

    def setup_patterns(self):
        """Configura padrões regex para validação"""
        self.keywords = {
            'begin', 'end', 'if', 'then', 'else', 'while', 'do', 'for',
            'procedure', 'function', 'unit', 'interface', 'implementation',
            'var', 'const', 'type', 'class', 'record', 'array', 'of',
            'program', 'uses', 'initialization', 'finalization', 'try',
            'except', 'finally', 'raise', 'private', 'protected', 'public',
            'published', 'property', 'constructor', 'destructor', 'inherited',
            'override', 'virtual', 'abstract', 'sealed', 'static'
        }

        self.patterns = {
            'unit_decl': re.compile(r'unit\s+([a-zA-Z][\w\.]*)\s*;'),
            'procedure_decl': re.compile(r'procedure\s+([a-zA-Z]\w*)\s*(\(.*?\))?\s*;'),
            'function_decl': re.compile(r'function\s+([a-zA-Z]\w*)\s*(\(.*?\))?\s*:\s*(\w+)\s*;'),
            'class_decl': re.compile(r'(\w+)\s*=\s*class\s*(\(.*?\))?'),
            'var_decl': re.compile(r'var\s+([^;]+);'),
            'const_decl': re.compile(r'const\s+([^;]+);'),
            'type_decl': re.compile(r'type\s+([^;]+);'),
            'uses_clause': re.compile(r'uses\s+([^;]+);'),
            'property_decl': re.compile(r'property\s+(\w+)\s*:\s*(\w+)\s*(?:read\s+\w+)?\s*(?:write\s+\w+)?\s*;'),
            'begin_end': re.compile(r'\b(begin|end)\b', re.IGNORECASE),
            'try_except': re.compile(r'\b(try|except|finally)\b', re.IGNORECASE)
        }

    def validate_syntax(self, content: str, file_path: Optional[str] = None) -> Tuple[bool, List[SyntaxError]]:
        """Valida a sintaxe do código Delphi"""
        errors = []
        lines = content.split('\n')

        # Verifica estrutura básica da unit
        if not self._check_basic_structure(content):
            errors.append(SyntaxError(
                line=1,
                column=1,
                message="Estrutura básica da unit inválida",
                severity=5,
                error_type="structural",
                suggestion="Verifique se a unit possui as seções interface e implementation"
            ))

        # Stack para controle de blocos
        block_stack = []
        in_string = False
        string_char = None

        for line_num, line in enumerate(lines, 1):
            # Processa caracteres da linha
            for col, char in enumerate(line, 1):
                # Verifica strings
                if char in ["'", '"']:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None

                if not in_string:
                    # Verifica blocos begin/end
                    if re.match(r'\bbegin\b', line[col - 1:], re.IGNORECASE):
                        block_stack.append(('begin', line_num, col))
                    elif re.match(r'\bend\b', line[col - 1:], re.IGNORECASE):
                        if not block_stack:
                            errors.append(SyntaxError(
                                line=line_num,
                                column=col,
                                message="'end' sem 'begin' correspondente",
                                severity=4,
                                error_type="block_mismatch",
                                suggestion="Adicione o 'begin' correspondente"
                            ))
                        else:
                            block_stack.pop()

            # Validações específicas por linha
            errors.extend(self._validate_line(line, line_num))

        # Verifica blocos não fechados
        for block_type, block_line, block_col in block_stack:
            errors.append(SyntaxError(
                line=block_line,
                column=block_col,
                message=f"Bloco '{block_type}' não foi fechado",
                severity=4,
                error_type="unclosed_block",
                suggestion=f"Adicione 'end' para fechar o bloco iniciado na linha {block_line}"
            ))

        return len(errors) == 0, errors

    def _check_basic_structure(self, content: str) -> bool:
        """Verifica estrutura básica de uma unit Delphi"""
        # Verifica declaração da unit
        unit_match = self.patterns['unit_decl'].search(content)
        if not unit_match:
            return False

        # Verifica seções obrigatórias
        has_interface = 'interface' in content.lower()
        has_implementation = 'implementation' in content.lower()

        # Verifica ordem das seções
        if has_interface and has_implementation:
            interface_pos = content.lower().find('interface')
            implementation_pos = content.lower().find('implementation')
            if implementation_pos < interface_pos:
                return False

        # Verifica final da unit
        ends_properly = content.strip().lower().endswith('end.')

        return has_interface and has_implementation and ends_properly

    def _validate_line(self, line: str, line_num: int) -> List[SyntaxError]:
        """Valida uma linha específica de código"""
        errors = []
        line = line.strip()

        if not line:
            return errors

        # Ignora comentários
        if line.startswith('//') or line.startswith('{'):
            return errors

        # Validações específicas por tipo de declaração
        if 'procedure' in line:
            if not self.patterns['procedure_decl'].match(line):
                errors.append(SyntaxError(
                    line=line_num,
                    column=1,
                    message="Declaração de procedure inválida",
                    severity=3,
                    error_type="invalid_declaration",
                    suggestion="Verifique a sintaxe da declaração do procedure"
                ))

        elif 'function' in line:
            if not self.patterns['function_decl'].match(line):
                errors.append(SyntaxError(
                    line=line_num,
                    column=1,
                    message="Declaração de function inválida",
                    severity=3,
                    error_type="invalid_declaration",
                    suggestion="Verifique a sintaxe da declaração da function"
                ))

        # Verifica ponto e vírgula
        if (not any(line.startswith(kw) for kw in ['begin', 'end', 'try', 'except', 'finally']) and
                not line.endswith(('begin', 'end', 'end;', 'end.')) and
                not line.endswith(';')):
            errors.append(SyntaxError(
                line=line_num,
                column=len(line),
                message="Falta ponto e vírgula no final da linha",
                severity=2,
                error_type="missing_semicolon",
                suggestion="Adicione ; no final da linha"
            ))

        return errors

    def validate_file(self, file_path: Path) -> Tuple[bool, List[SyntaxError]]:
        """Valida um arquivo Delphi completo"""
        try:
            # Detecta encoding
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                result = chardet.detect(raw_content)
                encoding = result['encoding'] or 'utf-8'

            # Lê conteúdo
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            return self.validate_syntax(content, str(file_path))

        except Exception as e:
            self.logger.error(f"Erro ao validar arquivo {file_path}: {str(e)}")
            return False, [SyntaxError(
                line=1,
                column=1,
                message=f"Erro ao processar arquivo: {str(e)}",
                severity=5,
                error_type="processing_error"
            )]