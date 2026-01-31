#!/usr/bin/env python3
"""
Скрипт для очистки БД и Qdrant (сброс к начальному состоянию).

Очищает:
- Interactions (все взаимодействия пользователей)
- User channel preference vectors и user channel tastes
- Taste clusters (кластеры вкусов)
- Qdrant collection (эмбеддинги постов)
"""

import asyncio
import sys
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Импорты моделей
from app.models.interaction import Interaction
from app.models.user_channel_preference_vector import UserChannelPreferenceVector
from app.models.user_channel_taste import UserChannelTaste
from app.models.taste_cluster import TasteCluster
from app.models.user import User
from app.models.channel import Channel
from app.models.post import Post
from app.models.user_channel import UserChannel
from app.config import get_settings

settings = get_settings()


async def clear_database():
    """Очистить ВСЕ таблицы в БД."""
    engine = create_async_engine(
        settings.database_url,
        echo=False,
    )
    
    async with AsyncSession(engine, expire_on_commit=False) as session:
        print("Очистка БД (все таблицы)...")
        
        # Удаляем все данные из всех таблиц (в правильном порядке из-за foreign keys)
        await session.execute(delete(Interaction))
        print("  ✓ Interactions удалены")
        
        await session.execute(delete(UserChannelTaste))
        print("  ✓ User channel tastes удалены")
        
        await session.execute(delete(UserChannelPreferenceVector))
        print("  ✓ User channel preference vectors удалены")

        await session.execute(delete(UserChannel))
        print("  ✓ User channels удалены")
        
        await session.execute(delete(Post))
        print("  ✓ Posts удалены")
        
        await session.execute(delete(Channel))
        print("  ✓ Channels удалены")
        
        await session.execute(delete(TasteCluster))
        print("  ✓ Taste clusters удалены")
        
        await session.execute(delete(User))
        print("  ✓ Users удалены")
        
        await session.commit()
    
    await engine.dispose()
    print("БД полностью очищена\n")


def clear_qdrant():
    """Очистить Qdrant collection."""
    print("Очистка Qdrant...")
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=settings.qdrant_timeout,
        )
        
        # Удаляем collection если существует
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if settings.qdrant_collection_name in collection_names:
            client.delete_collection(settings.qdrant_collection_name)
            print(f"  ✓ Collection '{settings.qdrant_collection_name}' удалена")
        else:
            print(f"  - Collection '{settings.qdrant_collection_name}' не найдена")
        
        # Пересоздаём пустую collection
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"  ✓ Collection '{settings.qdrant_collection_name}' создана заново")
        print("Qdrant очищен\n")
        
    except Exception as e:
        print(f"  ✗ Ошибка при очистке Qdrant: {e}")
        raise


async def main():
    """Основная функция."""
    print("=" * 60)
    print("Очистка БД и Qdrant")
    print("=" * 60)
    print()
    
    try:
        await clear_database()
        clear_qdrant()
        print("=" * 60)
        print("✓ Очистка завершена успешно")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
