"""
Модуль для управления метаданными инструкций в SQLite
"""
import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from src.config import METADATA_DB


class MetadataManager:
    """Класс для работы с базой данных метаданных"""

    def __init__(self, db_path: str = METADATA_DB):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Получение подключения к БД"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Для доступа к колонкам по имени
        return conn

    # === Работа с инструкциями ===

    def add_instruction(
        self,
        instruction_id: str,
        doc_id: str,
        title: str,
        file_path: str,
        file_format: str,
        source_type: str,
        separator_index: Optional[int] = None,
        author: str = "Admin",
        tags: List[str] = None,
        images: List[Dict] = None
    ) -> bool:
        """
        Добавление новой инструкции в БД

        Args:
            instruction_id: UUID инструкции
            doc_id: ID документа
            title: Название инструкции
            file_path: Путь к файлу
            file_format: Формат файла (docx, md, txt)
            source_type: Тип источника (single_file, multi_instruction)
            separator_index: Индекс после разделителя (для multi)
            author: Автор
            tags: Список тегов
            images: Список изображений

        Returns:
            True если успешно добавлено
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Добавляем инструкцию
            cursor.execute('''
                INSERT INTO instructions (
                    id, doc_id, title, file_path, file_format,
                    source_type, separator_index, author
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                instruction_id, doc_id, title, file_path, file_format,
                source_type, separator_index, author
            ))

            # Добавляем теги
            if tags:
                self._add_tags_to_instruction(cursor, instruction_id, tags)

            # Добавляем изображения
            if images:
                self._add_images_to_instruction(cursor, instruction_id, images)

            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            print(f"Ошибка при добавлении инструкции: {e}")
            return False

        finally:
            conn.close()

    def _add_tags_to_instruction(
        self,
        cursor: sqlite3.Cursor,
        instruction_id: str,
        tags: List[str]
    ):
        """Добавление тегов к инструкции"""
        for tag_name in tags:
            # Получаем или создаём тег
            cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
            result = cursor.fetchone()

            if result:
                tag_id = result[0]
            else:
                cursor.execute('INSERT INTO tags (name) VALUES (?)', (tag_name,))
                tag_id = cursor.lastrowid

            # Связываем тег с инструкцией
            cursor.execute(
                'INSERT OR IGNORE INTO instruction_tags (instruction_id, tag_id) VALUES (?, ?)',
                (instruction_id, tag_id)
            )

    def _add_images_to_instruction(
        self,
        cursor: sqlite3.Cursor,
        instruction_id: str,
        images: List[Dict]
    ):
        """Добавление изображений к инструкции"""
        for img in images:
            cursor.execute('''
                INSERT INTO instruction_images (
                    instruction_id, image_path, image_index, placeholder
                )
                VALUES (?, ?, ?, ?)
            ''', (
                instruction_id,
                img.get('path'),
                img.get('index'),
                img.get('placeholder')
            ))

    def get_instruction(self, instruction_id: str) -> Optional[Dict]:
        """Получение инструкции по ID"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM instructions WHERE id = ?', (instruction_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        instruction = dict(row)

        # Получаем теги
        cursor.execute('''
            SELECT t.name
            FROM tags t
            JOIN instruction_tags it ON t.id = it.tag_id
            WHERE it.instruction_id = ?
        ''', (instruction_id,))
        instruction['tags'] = [row[0] for row in cursor.fetchall()]

        # Получаем изображения
        cursor.execute('''
            SELECT image_path, image_index, placeholder
            FROM instruction_images
            WHERE instruction_id = ?
            ORDER BY image_index
        ''', (instruction_id,))
        instruction['images'] = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return instruction

    def get_all_instructions(self, active_only: bool = True) -> List[Dict]:
        """Получение всех инструкций"""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = 'SELECT * FROM instructions'
        if active_only:
            query += ' WHERE active = 1'
        query += ' ORDER BY created_at DESC'

        cursor.execute(query)
        instructions = [dict(row) for row in cursor.fetchall()]

        # Добавляем теги к каждой инструкции
        for inst in instructions:
            cursor.execute('''
                SELECT t.name
                FROM tags t
                JOIN instruction_tags it ON t.id = it.tag_id
                WHERE it.instruction_id = ?
            ''', (inst['id'],))
            inst['tags'] = [row[0] for row in cursor.fetchall()]

        conn.close()
        return instructions

    def get_instructions_by_tag(self, tag_name: str) -> List[Dict]:
        """Получение инструкций по тегу"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT i.*
            FROM instructions i
            JOIN instruction_tags it ON i.id = it.instruction_id
            JOIN tags t ON it.tag_id = t.id
            WHERE t.name = ? AND i.active = 1
            ORDER BY i.created_at DESC
        ''', (tag_name,))

        instructions = [dict(row) for row in cursor.fetchall()]

        # Добавляем теги
        for inst in instructions:
            cursor.execute('''
                SELECT t.name
                FROM tags t
                JOIN instruction_tags it ON t.id = it.tag_id
                WHERE it.instruction_id = ?
            ''', (inst['id'],))
            inst['tags'] = [row[0] for row in cursor.fetchall()]

        conn.close()
        return instructions

    def mark_instruction_inactive(self, instruction_id: str) -> bool:
        """Пометить инструкцию как неактуальную"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                'UPDATE instructions SET active = 0, updated_at = ? WHERE id = ?',
                (datetime.now().isoformat(), instruction_id)
            )
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Ошибка при пометке инструкции как неактуальной: {e}")
            return False
        finally:
            conn.close()

    def delete_instruction(self, instruction_id: str) -> bool:
        """Удаление инструкции (каскадно удалятся теги и изображения)"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM instructions WHERE id = ?', (instruction_id,))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Ошибка при удалении инструкции: {e}")
            return False
        finally:
            conn.close()

    # === Работа с тегами ===

    def get_all_tags(self) -> List[str]:
        """Получение всех тегов"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT name FROM tags ORDER BY name')
        tags = [row[0] for row in cursor.fetchall()]

        conn.close()
        return tags

    def add_tag(self, tag_name: str, category: str = None) -> bool:
        """Добавление нового тега"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                'INSERT OR IGNORE INTO tags (name, category) VALUES (?, ?)',
                (tag_name, category)
            )
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Ошибка при добавлении тега: {e}")
            return False
        finally:
            conn.close()

    # === Статистика ===

    def get_stats(self) -> Dict:
        """Получение статистики по базе"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM instructions WHERE active = 1')
        active_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM instructions WHERE active = 0')
        inactive_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM tags')
        tags_count = cursor.fetchone()[0]

        conn.close()

        return {
            'active_instructions': active_count,
            'inactive_instructions': inactive_count,
            'total_instructions': active_count + inactive_count,
            'total_tags': tags_count
        }

    def clear_all_data(self) -> bool:
        """
        Полная очистка всех данных из базы метаданных
        ВНИМАНИЕ: Это действие необратимо!

        Returns:
            bool: True если успешно, False при ошибке
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Удаляем все данные из таблиц (порядок важен из-за внешних ключей)
            cursor.execute('DELETE FROM instruction_images')
            cursor.execute('DELETE FROM instruction_tags')
            cursor.execute('DELETE FROM instruction_history')
            cursor.execute('DELETE FROM instructions')
            # Не удаляем теги, чтобы они остались доступны для новых инструкций
            # cursor.execute('DELETE FROM tags')

            conn.commit()
            print("✅ Все данные успешно удалены из базы метаданных")
            return True
        except Exception as e:
            conn.rollback()
            print(f"❌ Ошибка при очистке базы метаданных: {e}")
            return False
        finally:
            conn.close()
