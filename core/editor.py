import asyncio
import difflib
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, AsyncGenerator

import aiofiles
import chardet

from core.models import CodeChange
from core.syntax_validator import DelphiSyntaxValidator


class EditorError(Exception):
    """Exceção customizada para erros do editor"""
    pass


class BackupManager:
    """Gerenciador de backups de código"""

    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    async def create_backup(self, file_path: Path, metadata: Optional[Dict] = None) -> Path:
        """Cria backup com metadados"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}.bak"
            backup_path = self.backup_dir / backup_name

            # Cria backup de forma segura
            async with aiofiles.open(file_path, 'rb') as source:
                content = await source.read()

                # Usa arquivo temporário primeiro
                temp_fd, temp_path = tempfile.mkstemp(suffix='.bak')
                os.close(temp_fd)

                async with aiofiles.open(temp_path, 'wb') as temp:
                    await temp.write(content)

                # Move para localização final
                shutil.move(temp_path, backup_path)

            # Salva metadados
            meta = {
                'original_file': str(file_path),
                'timestamp': timestamp,
                'content_hash': hashlib.sha256(content).hexdigest(),
                'size': len(content)
            }
            if metadata:
                meta.update(metadata)

            meta_path = backup_path.with_suffix('.meta')
            async with aiofiles.open(meta_path, 'w') as f:
                await f.write(json.dumps(meta, indent=2))

            return backup_path

        except Exception as e:
            self.logger.error(f"Erro ao criar backup: {str(e)}")
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
            raise EditorError(f"Erro ao criar backup: {str(e)}")

    async def restore_backup(self, backup_path: Path, target_path: Path) -> bool:
        """Restaura um backup de forma segura"""
        try:
            # Verifica metadados primeiro
            meta_path = backup_path.with_suffix('.meta')
            if not meta_path.exists():
                raise EditorError("Metadados do backup não encontrados")

            async with aiofiles.open(meta_path, 'r') as f:
                metadata = json.loads(await f.read())

            if Path(metadata['original_file']) != target_path:
                raise EditorError("Backup não corresponde ao arquivo alvo")

            # Cria backup do estado atual antes de restaurar
            await self.create_backup(target_path, {'reason': 'pre_restore'})

            # Restaura usando arquivo temporário
            temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp')
            os.close(temp_fd)

            shutil.copy2(backup_path, temp_path)
            shutil.move(temp_path, target_path)

            return True

        except Exception as e:
            self.logger.error(f"Erro ao restaurar backup: {str(e)}")
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
            return False

    async def list_backups(self, file_path: Path) -> List[Dict]:
        """Lista backups disponíveis"""
        try:
            backups = []
            async for backup in self._scan_backups(file_path):
                backups.append(backup)
            return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            self.logger.error(f"Erro ao listar backups: {str(e)}")
            return []

    async def _scan_backups(self, file_path: Path) -> AsyncGenerator[Dict, None]:
        """Scanner assíncrono de backups"""
        for backup in self.backup_dir.glob(f"{file_path.stem}_*.bak"):
            meta_path = backup.with_suffix('.meta')
            if meta_path.exists():
                async with aiofiles.open(meta_path, 'r') as f:
                    metadata = json.loads(await f.read())
                    metadata['backup_path'] = str(backup)
                    yield metadata


class DelphiCodeEditor:
    """Editor de código Delphi com suporte a backups e rollback"""

    def __init__(self, backup_dir: str = "backups"):
        self.backup_manager = BackupManager(Path(backup_dir))
        self.validator = DelphiSyntaxValidator()
        self.logger = logging.getLogger(__name__)

        # Semáforo para controle de concorrência
        self._file_locks = {}
        self._lock_creation_lock = asyncio.Lock()

    async def _get_file_lock(self, file_path: str) -> asyncio.Lock:
        """Obtém lock específico para um arquivo"""
        async with self._lock_creation_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = asyncio.Lock()
            return self._file_locks[file_path]

    async def apply_changes(self, file_path: Path, changes: List[Dict[str, any]],
                            create_backup: bool = True) -> Tuple[bool, str]:
        """Aplica mudanças ao código com validação"""
        file_lock = await self._get_file_lock(str(file_path))
        async with file_lock:
            try:
                # Lê arquivo original
                content = await self._read_file_with_encoding(file_path)

                # Cria backup se necessário
                backup_path = None
                if create_backup:
                    backup_path = await self.backup_manager.create_backup(
                        file_path,
                        {'changes': changes}
                    )

                # Aplica mudanças
                new_content = content
                affected_lines = []

                for change in sorted(changes, key=lambda x: x.get('line', 0)):
                    line_num = change.get('line', 0)
                    old_code = change.get('old_code', '')
                    new_code = change.get('new_code', '')

                    if old_code not in content:
                        raise EditorError(f"Código original não encontrado: {old_code[:50]}...")

                    new_content = new_content.replace(old_code, new_code)
                    affected_lines.extend(range(
                        line_num,
                        line_num + len(old_code.splitlines())
                    ))

                # Valida novo código
                valid, errors = await self.validator.validate_syntax(new_content)
                if not valid:
                    if backup_path:
                        await self.backup_manager.restore_backup(backup_path, file_path)
                    return False, f"Erros de sintaxe: {', '.join(str(e) for e in errors)}"

                # Salva mudanças usando arquivo temporário
                temp_fd, temp_path = tempfile.mkstemp(suffix='.pas')
                os.close(temp_fd)

                async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                    await f.write(new_content)

                shutil.move(temp_path, file_path)

                return True, str(backup_path) if backup_path else ""

            except Exception as e:
                self.logger.error(f"Erro ao aplicar mudanças: {str(e)}")
                if backup_path:
                    await self.backup_manager.restore_backup(backup_path, file_path)
                if 'temp_path' in locals() and Path(temp_path).exists():
                    Path(temp_path).unlink()
                return False, str(e)

    async def apply_change(self, change: CodeChange) -> Tuple[bool, Optional[str]]:
        """Aplica uma única mudança com validação"""
        try:
            # Valida mudança
            valid, errors = await self.validator.validate_syntax(change.new_content)
            if not valid:
                return False, f"Validação falhou: {', '.join(errors)}"

            # Aplica mudança usando arquivo temporário
            file_path = Path(change.file_path)
            backup_path = await self.backup_manager.create_backup(file_path)

            temp_fd, temp_path = tempfile.mkstemp(suffix='.pas')
            os.close(temp_fd)

            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            modified_content = content.replace(
                change.original_content,
                change.new_content
            )

            async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                await f.write(modified_content)

            shutil.move(temp_path, file_path)

            return True, str(backup_path)

        except Exception as e:
            self.logger.error(f"Erro ao aplicar mudança: {str(e)}")
            if 'backup_path' in locals():
                await self.backup_manager.restore_backup(Path(backup_path), file_path)
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
            return False, str(e)

    async def compare_versions(self, original_path: Path, modified_path: Path) -> List[Dict[str, any]]:
        """Compara duas versões do código e retorna as diferenças"""
        try:
            # Lê arquivos
            original_content = await self._read_file_with_encoding(original_path)
            modified_content = await self._read_file_with_encoding(modified_path)

            # Gera diff
            original_lines = original_content.splitlines()
            modified_lines = modified_content.splitlines()

            differ = difflib.Differ()
            diff = list(differ.compare(original_lines, modified_lines))

            # Processa diferenças
            changes = []
            current_change = None

            for line_num, line in enumerate(diff, 1):
                if line.startswith('- '):
                    if current_change:
                        changes.append(current_change)
                    current_change = {
                        'type': 'removal',
                        'start_line': line_num,
                        'end_line': line_num,
                        'old_code': line[2:],
                        'new_code': ''
                    }
                elif line.startswith('+ '):
                    if current_change and current_change['type'] == 'removal':
                        current_change['type'] = 'modification'
                        current_change['new_code'] = line[2:]
                        changes.append(current_change)
                        current_change = None
                    else:
                        if current_change:
                            changes.append(current_change)
                        current_change = {
                            'type': 'addition',
                            'start_line': line_num,
                            'end_line': line_num,
                            'old_code': '',
                            'new_code': line[2:]
                        }
                elif line.startswith('? '):
                    continue
                else:  # Linha sem mudança
                    if current_change:
                        changes.append(current_change)
                        current_change = None

            if current_change:
                changes.append(current_change)

            return changes

        except Exception as e:
            self.logger.error(f"Erro ao comparar versões: {str(e)}")
            return []

    async def validate_changes(self, changes: List[Dict[str, any]], file_path: Path) -> Tuple[bool, List[str]]:
        """Valida um conjunto de mudanças antes de aplicá-las"""
        try:
            # Lê conteúdo atual
            content = await self._read_file_with_encoding(file_path)

            # Aplica mudanças em memória
            modified_content = content
            for change in sorted(changes, key=lambda x: x.get('start_line', 0)):
                if change['type'] in ['modification', 'removal']:
                    if change['old_code'] not in modified_content:
                        return False, [f"Código original não encontrado: {change['old_code'][:50]}..."]
                    modified_content = modified_content.replace(
                        change['old_code'],
                        change.get('new_code', '')
                    )
                elif change['type'] == 'addition':
                    lines = modified_content.splitlines()
                    if 0 <= change['start_line'] <= len(lines):
                        lines.insert(change['start_line'], change['new_code'])
                        modified_content = '\n'.join(lines)
                    else:
                        return False, [f"Linha inválida para adição: {change['start_line']}"]

            # Valida sintaxe
            valid, errors = await self.validator.validate_syntax(modified_content)
            if not valid:
                return False, [str(error) for error in errors]

            return True, []

        except Exception as e:
            self.logger.error(f"Erro ao validar mudanças: {str(e)}")
            return False, [str(e)]

    async def _read_file_with_encoding(self, file_path: Path) -> str:
        """Lê arquivo detectando codificação automaticamente"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                raw_data = await f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'

            async with aiofiles.open(file_path, encoding=encoding) as f:
                return await f.read()

        except Exception as e:
            self.logger.error(f"Erro ao ler arquivo {file_path}: {str(e)}")
            raise EditorError(f"Erro ao ler arquivo: {str(e)}")