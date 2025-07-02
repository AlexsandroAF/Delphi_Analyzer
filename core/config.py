import base64
import logging
import secrets
from dataclasses import dataclass, asdict, field
from hashlib import sha256
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


@dataclass
class DatabaseConfig:
    """Configurações do banco de dados"""
    path: str
    backup_retention_days: int = 7
    cache_enabled: bool = True
    cache_ttl_days: int = 1
    max_connections: int = 5
    connection_timeout: int = 30
    max_pool_size: int = 10


@dataclass
class ApiConfig:
    """Configurações da API"""
    key: str = field(repr=False)  # Não mostra a chave em logs/repr
    model: str = "claude-3-haiku-20240307"
    max_tokens: int = 4000
    temperature: float = 0
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_tokens: int = 50000


@dataclass
class AnalysisConfig:
    """Configurações de análise"""
    min_severity: int = 1
    auto_fix_enabled: bool = False
    auto_fix_min_severity: int = 4
    monitor_interval: int = 300
    ignore_patterns: list = field(default_factory=lambda: ['*.bak', '*.~pas'])
    parallel_analysis: bool = True
    max_parallel_files: int = 4


@dataclass
class SecurityConfig:
    """Configurações de segurança"""
    encryption_enabled: bool = True
    key_rotation_days: int = 30
    min_key_length: int = 32
    hash_algorithm: str = 'sha256'
    salt_length: int = 16


@dataclass
class SystemConfig:
    """Configuração completa do sistema"""
    database: DatabaseConfig
    api: ApiConfig
    analysis: AnalysisConfig
    security: SecurityConfig
    project_dir: str
    backup_dir: str
    log_dir: str
    debug: bool = False
    environment: str = 'development'


class SimpleEncryption:
    """Implementação simples de criptografia"""

    def __init__(self):
        self.key_path = Path("../.env.key")
        self._load_or_create_key()

    def _load_or_create_key(self):
        if not self.key_path.exists():
            key = secrets.token_bytes(32)
            self.key_path.write_bytes(key)
        self._key = self.key_path.read_bytes()

    def encrypt(self, data: str) -> str:
        if not data:
            return ""
        try:
            key = sha256(self._key).digest()
            encrypted = bytes([ord(c) ^ key[i % len(key)] for i, c in enumerate(data)])
            return f"ENC[{base64.b64encode(encrypted).decode()}]"
        except Exception as e:
            logging.error(f"Erro ao encriptar: {e}")
            return data

    def decrypt(self, data: str) -> str:
        if not data or not data.startswith("ENC["):
            return data
        try:
            encrypted = base64.b64decode(data[4:-1])
            key = sha256(self._key).digest()
            decrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(encrypted)])
            return decrypted.decode()
        except Exception as e:
            logging.error(f"Erro ao decriptar: {e}")
            return data


class ConfigurationError(Exception):
    """Exceção para erros de configuração"""
    pass


class ConfigManager:
    """Gerenciador de configurações do sistema"""

    def __init__(self, config_path: str = "config.yml"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self._config: Optional[SystemConfig] = None
        self.encryption = SimpleEncryption()

    def load_config(self) -> SystemConfig:
        """Carrega configurações do arquivo"""
        try:
            if not self.config_path.exists():
                self.logger.warning(f"Arquivo de configuração não encontrado: {self.config_path}")
                self._config = self._create_default_config()
                self.save_config()
                return self._config

            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Processa dados sensíveis
            api_data = data.get('api', {})
            api_key = api_data.get('key', '')
            if api_key.startswith('ENC['):
                api_key = self.encryption.decrypt(api_key)

            # Cria objetos de configuração
            self._config = SystemConfig(
                database=DatabaseConfig(**data.get('database', {})),
                api=ApiConfig(
                    key=api_key,
                    model=api_data.get('model', "claude-3-haiku-20240307"),
                    max_tokens=api_data.get('max_tokens', 4000),
                    temperature=api_data.get('temperature', 0),
                    timeout=api_data.get('timeout', 30),
                    max_retries=api_data.get('max_retries', 3),
                    retry_delay=api_data.get('retry_delay', 1.0),
                    rate_limit_tokens=api_data.get('rate_limit_tokens', 50000)
                ),
                analysis=AnalysisConfig(**data.get('analysis', {})),
                security=SecurityConfig(**data.get('security', {})),
                project_dir=data.get('project_dir', '.'),
                backup_dir=data.get('backup_dir', 'backups'),
                log_dir=data.get('log_dir', 'logs'),
                debug=data.get('debug', False),
                environment=data.get('environment', 'development')
            )

            self._validate_config(self._config)
            return self._config

        except Exception as e:
            self.logger.error(f"Erro ao carregar configurações: {str(e)}")
            raise ConfigurationError(f"Erro ao carregar configurações: {str(e)}")

    def save_config(self):
        """Salva configurações no arquivo"""
        if not self._config:
            raise ConfigurationError("Configurações não inicializadas")

        try:
            config_dict = {
                'database': asdict(self._config.database),
                'api': {
                    'key': self.encryption.encrypt(self._config.api.key),
                    'model': self._config.api.model,
                    'max_tokens': self._config.api.max_tokens,
                    'temperature': self._config.api.temperature,
                    'timeout': self._config.api.timeout,
                    'max_retries': self._config.api.max_retries,
                    'retry_delay': self._config.api.retry_delay,
                    'rate_limit_tokens': self._config.api.rate_limit_tokens
                },
                'analysis': asdict(self._config.analysis),
                'security': asdict(self._config.security),
                'project_dir': self._config.project_dir,
                'backup_dir': self._config.backup_dir,
                'log_dir': self._config.log_dir,
                'debug': self._config.debug,
                'environment': self._config.environment
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)

        except Exception as e:
            self.logger.error(f"Erro ao salvar configurações: {str(e)}")
            raise ConfigurationError(f"Erro ao salvar configurações: {str(e)}")

    def _create_default_config(self) -> SystemConfig:
        """Cria configuração padrão"""
        return SystemConfig(
            database=DatabaseConfig(
                path="../analysis.db",
                backup_retention_days=7,
                cache_enabled=True,
                cache_ttl_days=1,
                max_connections=5,
                connection_timeout=30,
                max_pool_size=10
            ),
            api=ApiConfig(
                key="",  # Deve ser preenchido pelo usuário
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0,
                timeout=30,
                max_retries=3,
                retry_delay=1.0,
                rate_limit_tokens=50000
            ),
            analysis=AnalysisConfig(
                min_severity=1,
                auto_fix_enabled=False,
                auto_fix_min_severity=4,
                monitor_interval=300,
                ignore_patterns=['*.bak', '*.~pas', '*.dcu', '*.tmp'],
                parallel_analysis=True,
                max_parallel_files=4
            ),
            security=SecurityConfig(),
            project_dir="..",
            backup_dir="../backups",
            log_dir="logs",
            debug=False,
            environment='development'
        )

    def _validate_config(self, config: SystemConfig):
        """Valida configuração"""
        if not config.api.key:
            raise ConfigurationError("Chave da API não configurada")

        if config.api.max_tokens < 1 or config.api.max_tokens > 100000:
            raise ConfigurationError("max_tokens deve estar entre 1 e 100000")

        if config.api.temperature < 0 or config.api.temperature > 1:
            raise ConfigurationError("temperature deve estar entre 0 e 1")

    def get_config(self) -> SystemConfig:
        """Retorna configuração atual"""
        if not self._config:
            return self.load_config()
        return self._config

    def update_config(self, updates: Dict[str, Any]):
        """Atualiza configurações específicas"""
        if not self._config:
            self.load_config()

        for section, values in updates.items():
            if hasattr(self._config, section):
                section_config = getattr(self._config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        self.save_config()