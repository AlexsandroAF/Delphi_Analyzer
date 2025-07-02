from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class DelphiToken:
    """Representa um token da linguagem Delphi"""
    type: str
    value: str
    line: int
    column: int
    file_path: Optional[str] = None


@dataclass
class CodeBlock:
    """Representa um bloco de código Delphi"""
    type: str  # Tipo do bloco (procedure, function, etc)
    name: str  # Nome do bloco
    content: str  # Conteúdo completo
    start_line: int
    end_line: int
    tokens: List[DelphiToken] = field(default_factory=list)
    parent: Optional['CodeBlock'] = None
    children: List['CodeBlock'] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class CodeChange:
    """Representa uma mudança no código"""
    file_path: str
    original_content: str
    new_content: str
    change_type: str  # 'fix', 'refactor', 'security', etc
    timestamp: datetime
    description: str
    priority: int  # 1-5
    affected_lines: List[int]
    backup_path: Optional[str] = None
    author: str = "AI_Assistant"
    validated: bool = False
    applied: bool = False


@dataclass
class AnalysisResult:
    """Resultado da análise de código"""
    file_path: str
    timestamp: datetime
    issues: List[dict]
    metrics: dict
    suggestions: List[dict]
    priority: int  # 1-5
    content_hash: str
    execution_time: float


@dataclass
class ApiInteraction:
    """Registra uma interação com a API"""
    timestamp: datetime
    prompt: str
    response: str
    tokens_used: int
    cost: float
    success: bool
    model: str = "claude-3-haiku-20240307"
    error: Optional[str] = None
    latency: Optional[float] = None


@dataclass
class DelphiFile:
    """Representa um arquivo Delphi completo"""
    path: Path
    content: str
    encoding: str
    unit_name: str
    interface_section: Optional[CodeBlock] = None
    implementation_section: Optional[CodeBlock] = None
    initialization_section: Optional[CodeBlock] = None
    finalization_section: Optional[CodeBlock] = None
    uses_clauses: List[str] = field(default_factory=list)
    declarations: List[CodeBlock] = field(default_factory=list)
    last_modified: Optional[datetime] = None
    content_hash: Optional[str] = None