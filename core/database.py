import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import aiosqlite

from core.models import AnalysisResult, ApiInteraction


class ConnectionPool:
    """Pool de conexões para o banco de dados"""

    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connections = asyncio.Queue()
        self._active_connections = 0
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)  # Adiciona logger

    async def get_connection(self) -> aiosqlite.Connection:
        """Obtém uma conexão do pool"""
        async with self._lock:
            if self._active_connections < self.max_connections and self._connections.empty():
                self.logger.debug(f"Criando nova conexão. Ativas: {self._active_connections}")
                conn = await aiosqlite.connect(self.db_path)
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
                self._active_connections += 1
                return conn

        self.logger.debug("Aguardando conexão disponível do pool")
        return await self._connections.get()

    async def release_connection(self, conn: aiosqlite.Connection):
        """Devolve uma conexão ao pool"""
        self.logger.debug("Devolvendo conexão ao pool")
        await self._connections.put(conn)

    async def close_all(self):
        """Fecha todas as conexões"""
        while not self._connections.empty():
            conn = await self._connections.get()
            await conn.close()
        self._active_connections = 0


class Cache:
    """Cache em memória com TTL"""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache"""
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if (datetime.now() - timestamp).seconds < self._ttl:
                    return value
                del self._cache[key]
        return None

    async def set(self, key: str, value: Any):
        """Define valor no cache"""
        async with self._lock:
            self._cache[key] = (value, datetime.now())

    async def clear(self):
        """Limpa cache expirado"""
        async with self._lock:
            current_time = datetime.now()
            expired = [
                k for k, (_, ts) in self._cache.items()
                if (current_time - ts).seconds >= self._ttl
            ]
            for k in expired:
                del self._cache[k]


class DatabaseError(Exception):
    """Exceção customizada para erros de banco"""
    pass


class CodeAnalysisDatabase:
    """Gerencia o banco de dados de análises de código"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.pool = ConnectionPool(db_path)
        self.cache = Cache()
        self._migration_lock = asyncio.Lock()
        self._in_transaction = asyncio.Lock()

    @classmethod
    async def create(cls, db_path: str):
        """Método de fábrica para criar uma instância e configurar o banco"""
        instance = cls(db_path)
        await instance.setup_database()
        return instance

    @asynccontextmanager
    async def transaction(self):
        """Gerenciador de contexto para transações com suporte a savepoints."""
        conn = await self.pool.get_connection()
        savepoint_name = f"sp_{uuid.uuid4().hex[:8]}"  # Nome único para o savepoint

        try:
            if self._in_transaction.locked():
                # Transação já em andamento: cria um savepoint
                self.logger.debug(f"Criando savepoint: {savepoint_name}")
                async with conn.cursor() as cursor:
                    await cursor.execute(f"SAVEPOINT {savepoint_name}")
                    yield cursor
                    await cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            else:
                # Transação principal
                async with self._in_transaction:
                    async with conn.cursor() as cursor:
                        await cursor.execute("BEGIN")
                        yield cursor
                        await cursor.execute("COMMIT")

        except Exception as e:
            self.logger.error(f"Erro na transação: {str(e)}")
            if self._in_transaction.locked():
                # Restaura o savepoint em caso de erro
                async with conn.cursor() as cursor:
                    await cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            else:
                # Restaura a transação principal
                async with conn.cursor() as cursor:
                    await cursor.execute("ROLLBACK")
            raise DatabaseError(f"Erro na transação: {str(e)}")

        finally:
            await self.pool.release_connection(conn)

    async def setup_database(self):
        """Configura o esquema do banco de dados"""
        async with self._migration_lock:
            async with self.transaction() as cursor:
                # Tabelas principais
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analyses (
                        id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        content_hash TEXT NOT NULL,
                        issues TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        suggestions TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        execution_time REAL NOT NULL,
                        UNIQUE(file_path, content_hash)
                    )
                """)

                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS changes (
                        id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        change_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        original_content TEXT NOT NULL,
                        new_content TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        affected_lines TEXT NOT NULL,
                        backup_path TEXT,
                        author TEXT NOT NULL,
                        validated BOOLEAN NOT NULL DEFAULT 0,
                        applied BOOLEAN NOT NULL DEFAULT 0
                    )
                """)

                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_interactions (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        tokens_used INTEGER NOT NULL,
                        cost REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        model TEXT NOT NULL,
                        error TEXT,
                        latency REAL
                    )
                """)

                # Tabela de cache com índices otimizados
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_cache (
                        file_path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        last_analysis DATETIME NOT NULL,
                        cache_data TEXT NOT NULL,
                        PRIMARY KEY (file_path, content_hash)
                    )
                """)

                # Índices otimizados
                indices = [
                    "CREATE INDEX IF NOT EXISTS idx_analyses_file_path ON analyses(file_path)",
                    "CREATE INDEX IF NOT EXISTS idx_changes_file_path ON changes(file_path)",
                    "CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_interactions(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_cache_last_analysis ON analysis_cache(last_analysis)"
                ]

                for index in indices:
                    await cursor.execute(index)

    async def save_analysis(self, result: AnalysisResult) -> bool:
        """Salva resultado de análise no banco"""
        try:
            async with self.transaction() as cursor:
                await cursor.execute("""
                    INSERT INTO analyses 
                    (id, file_path, timestamp, content_hash, issues, 
                     metrics, suggestions, priority, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_path, content_hash) 
                    DO UPDATE SET
                        timestamp = excluded.timestamp,
                        issues = excluded.issues,
                        metrics = excluded.metrics,
                        suggestions = excluded.suggestions,
                        priority = excluded.priority,
                        execution_time = excluded.execution_time
                """, (
                    str(uuid.uuid4()),
                    result.file_path,
                    result.timestamp.isoformat(),
                    result.content_hash,
                    json.dumps(result.issues),
                    json.dumps(result.metrics),
                    json.dumps(result.suggestions),
                    result.priority,
                    result.execution_time
                ))
                return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar análise: {str(e)}")
            raise DatabaseError(f"Erro ao salvar análise: {str(e)}")

    async def save_changes(self, file_path: str, changes: List[Dict[str, Any]],
                           backup_path: Optional[str] = None) -> bool:
        """Salva mudanças aplicadas no código"""
        try:
            async with self.transaction() as cursor:
                for change in changes:
                    await cursor.execute("""
                        INSERT INTO changes 
                        (id, file_path, timestamp, change_type, description, 
                         original_content, new_content, priority, affected_lines,
                         backup_path, author, validated, applied)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(uuid.uuid4()),  # id
                        file_path,  # file_path
                        datetime.now().isoformat(),  # timestamp
                        change.get('type', 'fix'),  # change_type
                        change.get('description', ''),  # description
                        change.get('old_code', ''),  # original_content
                        change.get('new_code', ''),  # new_content
                        change.get('priority', 1),  # priority
                        json.dumps(change.get('affected_lines', [])),  # affected_lines
                        backup_path,  # backup_path
                        'AI_Assistant',  # author
                        True,  # validated
                        True  # applied
                    ))
                return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar mudanças: {str(e)}")
            return False

    async def save_api_interaction(self, interaction: ApiInteraction):
        """Salva interação com a API no banco de dados"""
        conn = await self.pool.get_connection()
        savepoint_name = f"sp_{uuid.uuid4().hex[:8]}"  # Nome único para o savepoint
        try:
            async with self.transaction() as cursor:
                await cursor.execute(f"SAVEPOINT {savepoint_name}")
                await cursor.execute("""
                    INSERT INTO api_interactions
                    (id, timestamp, prompt, response, tokens_used, cost, success, model, error, latency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    interaction.timestamp.isoformat(),
                    interaction.prompt,
                    interaction.response,
                    interaction.tokens_used,
                    interaction.cost,
                    interaction.success,
                    interaction.model,
                    interaction.error,
                    interaction.latency
                ))
                await cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                self.logger.info("Interação com a API salva com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao salvar interação com a API: {str(e)}")
            async with conn.cursor() as cursor:
                await cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            raise DatabaseError(f"Erro ao salvar interação com a API: {str(e)}")
        finally:
            await self.pool.release_connection(conn)


    async def get_cached_analysis(self, file_path: str, content_hash: str) -> Optional[Dict[str, Any]]:
        """Obtém análise em cache se disponível"""
        try:
            # Tenta cache em memória primeiro
            cache_key = f"{file_path}:{content_hash}"
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

            # Tenta cache no banco
            async with self.transaction() as cursor:
                await cursor.execute("""
                    SELECT cache_data 
                    FROM analysis_cache 
                    WHERE file_path = ? AND content_hash = ?
                    AND last_analysis > datetime('now', '-1 day')
                """, (file_path, content_hash))
                row = await cursor.fetchone()

                if row:
                    data = json.loads(row[0])
                    await self.cache.set(cache_key, data)
                    return data
                return None

        except Exception as e:
            self.logger.error(f"Erro ao obter cache: {str(e)}")
            return None

    async def save_to_cache(self, file_path: str, content_hash: str, data: Dict[str, Any]) -> bool:
        """Salva resultado no cache"""
        try:
            cache_key = f"{file_path}:{content_hash}"
            await self.cache.set(cache_key, data)

            async with self.transaction() as cursor:
                await cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (file_path, content_hash, last_analysis, cache_data)
                    VALUES (?, ?, ?, ?)
                """, (
                    file_path,
                    content_hash,
                    datetime.now().isoformat(),
                    json.dumps(data)
                ))
                return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar no cache: {str(e)}")
            return False

    async def cleanup_old_cache(self, days: int = 7) -> bool:
        """Remove entradas antigas do cache"""
        try:
            await self.cache.clear()

            async with self.transaction() as cursor:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                await cursor.execute("""
                    DELETE FROM analysis_cache 
                    WHERE last_analysis < ?
                """, (cutoff_date,))
                return True
        except Exception as e:
            self.logger.error(f"Erro ao limpar cache: {str(e)}")
            return False

    async def get_file_analysis_history(self, file_path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtém histórico de análises de um arquivo"""
        try:
            async with self.transaction() as cursor:
                cursor.row_factory = aiosqlite.Row
                await cursor.execute("""
                    SELECT * FROM analyses 
                    WHERE file_path = ? 
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (file_path, limit))
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Erro ao obter histórico de análises: {str(e)}")
            return []

    async def get_project_metrics(self) -> Dict[str, Any]:
        """Obtém métricas gerais do projeto"""
        try:
            async with self.transaction() as cursor:
                metrics = {}

                # Métricas de análise
                metrics.update(await self._get_analysis_metrics(cursor))

                # Métricas de mudanças
                metrics.update(await self._get_change_metrics(cursor))

                # Métricas de API
                metrics.update(await self._get_api_metrics(cursor))

                return metrics

        except Exception as e:
            self.logger.error(f"Erro ao obter métricas do projeto: {str(e)}")
            return {}

    async def _get_analysis_metrics(self, cursor) -> Dict[str, Any]:
        """Obtém métricas relacionadas a análises"""
        await cursor.execute("""
            SELECT 
                COUNT(DISTINCT file_path) as total_files,
                AVG(execution_time) as avg_execution_time,
                SUM(CASE WHEN priority >= 4 THEN 1 ELSE 0 END) as critical_issues
            FROM analyses
        """)
        row = await cursor.fetchone()

        return {
            'total_files_analyzed': row[0],
            'avg_execution_time': row[1],
            'critical_issues': row[2]
        }

    async def _get_change_metrics(self, cursor) -> Dict[str, Any]:
        """Obtém métricas relacionadas a mudanças"""
        await cursor.execute("""
            SELECT 
                COUNT(*) as total_changes,
                SUM(CASE WHEN applied = 1 THEN 1 ELSE 0 END) as applied_changes,
                AVG(priority) as avg_priority
            FROM changes
        """)
        row = await cursor.fetchone()

        return {
            'total_changes': row[0],
            'applied_changes': row[1],
            'avg_change_priority': row[2]
        }

    async def _get_api_metrics(self, cursor) -> Dict[str, Any]:
        """Obtém métricas relacionadas ao uso da API"""
        await cursor.execute("""
            SELECT 
                SUM(tokens_used) as total_tokens,
                SUM(cost) as total_cost,
                AVG(latency) as avg_latency,
                COUNT(*) as total_calls
            FROM api_interactions
            WHERE success = 1
        """)
        row = await cursor.fetchone()

        return {
            'api_total_tokens': row[0],
            'api_total_cost': row[1],
            'api_avg_latency': row[2],
            'api_total_calls': row[3]
        }

    async def close(self):
        """Fecha todas as conexões e limpa recursos"""
        await self.pool.close_all()
        await self.cache.clear()

    def __del__(self):
        """Garante limpeza de recursos na destruição do objeto"""
        try:
            asyncio.create_task(self.close())
        except:
            pass
