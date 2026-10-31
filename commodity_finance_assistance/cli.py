
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from commodity_finance_assistance.assistant_modes import (
    get_mode_instruction,
)
from commodity_finance_assistance.llm.service import LLMService
from commodity_finance_assistance.memory.database import (
    get_connection,
    initialize_database,
)
from commodity_finance_assistance.rag.embeddings import EmbeddingService
from commodity_finance_assistance.rag.ingestion import (
    RAGIngestionService,
)
from commodity_finance_assistance.rag.query import RAGQueryService
from commodity_finance_assistance.rag.store import RAGStore
from commodity_finance_assistance.tools.alpha_vantage import (
    AlphaVantageProvider,
)
from commodity_finance_assistance.tools.market_data import MarketDataTool
from commodity_finance_assistance.tools.mcx_bhavcopy import (
    MCXBhavcopyProvider,
)
from commodity_finance_assistance.tools.mcx_importer import (
    get_latest_quote,
    import_bhavcopy,
)
from commodity_finance_assistance.rag.evaluation import RAGEvaluator
from data.knowledge.evaluation_dataset import RAG_EVALUATION_CASES
from commodity_finance_assistance.hitl.console import (
    handle_approvals,
    handle_approve,
    handle_reject,
)


DEFAULT_KNOWLEDGE_DIRECTORY = Path("data/knowledge")


class LazyRAGQueryService:
    """Initialize embedding and query services only when RAG is requested."""

    def __init__(self, store: RAGStore) -> None:
        self.store = store
        self._query_service: RAGQueryService | None = None

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        if self._query_service is None:
            embedding_service = EmbeddingService()
            self._query_service = RAGQueryService(
                embedding_service=embedding_service,
                store=self.store,
            )

        return self._query_service.query(
            question,
            top_k=top_k,
            min_score=min_score,
        )


def print_conversations(service: LLMService) -> None:
    conversations = service.memory.list_conversations(limit=20)

    if not conversations:
        print("No saved conversations found.")
        return

    print("\nSaved conversations:")

    for item in conversations:
        title = item["title"] or "Untitled conversation"
        print(f"\n  Title: {title}")
        print(f"  ID: {item['conversation_id']}")
        print(f"  Messages: {item['message_count']}")
        print(f"  Created: {item['created_at']}")


def print_history(service: LLMService, conversation_id: str) -> None:
    messages = service.memory.get_messages(
        conversation_id,
        limit=100,
    )

    if not messages:
        print("No messages in this conversation yet.")
        return

    print("\nConversation history:")

    for message in messages:
        print(f"\n{message['role'].upper()}: {message['content']}")


def handle_quote(
    market_data: MarketDataTool,
    user_input: str,
) -> None:
    parts = user_input.split(maxsplit=1)

    if len(parts) != 2 or not parts[1].strip():
        print("Usage: /quote <commodity>")
        return

    try:
        quote = market_data.get_quote(parts[1])

        if quote is None:
            print(
                "No quote is available from the configured "
                "market-data provider."
            )
        else:
            print("\n" + market_data.format_quote(quote))

    except (ValueError, RuntimeError) as exc:
        print(f"Market data error: {exc}")


def handle_mcx_import(user_input: str) -> None:
    parts = user_input.split(maxsplit=1)

    csv_path = (
        parts[1].strip().strip('"')
        if len(parts) > 1
        else "data/mcx_bhavcopy.csv"
    )

    try:
        with get_connection() as connection:
            result = import_bhavcopy(csv_path, connection)

        print(
            "MCX import complete. "
            f"Rows read: {result.rows_read}; "
            f"processed: {result.rows_imported}; "
            f"skipped: {result.rows_skipped}."
        )

    except (
        ValueError,
        FileNotFoundError,
        OSError,
        sqlite3.Error,
    ) as exc:
        print(f"MCX import error: {exc}")


def handle_mcx(user_input: str) -> None:
    parts = user_input.split(maxsplit=1)

    if len(parts) != 2 or not parts[1].strip():
        print("Usage: /mcx <commodity>")
        return

    try:
        with get_connection() as connection:
            quote = get_latest_quote(connection, parts[1])

        if quote is None:
            print(
                "No matching MCX futures data found. "
                "Run /mcx-import first."
            )
        else:
            print("\n" + MCXBhavcopyProvider.format_quote(quote))

    except (ValueError, sqlite3.Error) as exc:
        print(f"MCX query error: {exc}")


def handle_ingest(
    user_input: str,
    ingestion_service: RAGIngestionService,
    store: RAGStore,
) -> None:
    """Ingest documents and generate persistent embeddings."""
    parts = user_input.split(maxsplit=1)

    directory = (
        Path(parts[1].strip().strip('"'))
        if len(parts) > 1
        else DEFAULT_KNOWLEDGE_DIRECTORY
    )

    if not directory.is_dir():
        print(f"Knowledge directory not found: {directory}")
        return

    try:
        print(f"\nScanning documents in: {directory}")

        results = ingestion_service.ingest_directory(directory)

        if not results:
            print("No supported documents found.")
            return

        # Create the embedding client only when ingestion is requested.
        embedding_service = EmbeddingService()

        total_documents = 0
        total_chunks = 0
        total_embeddings = 0

        for result in results:
            chunks = store.get_chunks(result.document_id)

            if not chunks:
                print(
                    f"Skipping embedding: no stored chunks for "
                    f"{result.title}"
                )
                continue

            vectors = embedding_service.embed_texts(
                [chunk["text"] for chunk in chunks]
            )

            embeddings = {
                chunk["chunk_id"]: vector
                for chunk, vector in zip(
                    chunks,
                    vectors,
                    strict=True,
                )
            }

            saved_count = store.save_embeddings(
                embeddings=embeddings,
                model=embedding_service.model,
                dimensions=embedding_service.dimensions,
            )

            total_documents += 1
            total_chunks += len(chunks)
            total_embeddings += saved_count

            print(
                f"Indexed: {result.title} | "
                f"Chunks: {len(chunks)} | "
                f"Embeddings: {saved_count}"
            )

        print("\nDocument ingestion completed.")
        print(f"Documents processed: {total_documents}")
        print(f"Chunks processed: {total_chunks}")
        print(f"Embeddings saved: {total_embeddings}")
        print(f"Embedding model: {embedding_service.model}")

    except Exception as exc:
        print(
            f"RAG ingestion error: {type(exc).__name__}: {exc}"
        )


def handle_rag_status(store: RAGStore) -> None:
    """Display the current RAG database statistics."""
    try:
        with get_connection() as connection:
            documents = connection.execute(
                "SELECT COUNT(*) FROM rag_documents"
            ).fetchone()[0]

            chunks = connection.execute(
                "SELECT COUNT(*) FROM rag_chunks"
            ).fetchone()[0]

            embeddings = connection.execute(
                "SELECT COUNT(*) FROM rag_embeddings"
            ).fetchone()[0]

            models = connection.execute(
                """
                SELECT DISTINCT model
                FROM rag_embeddings
                ORDER BY model
                """
            ).fetchall()

        print("\nRAG Knowledge Base Status")
        print(f"Documents: {documents}")
        print(f"Chunks: {chunks}")
        print(f"Embeddings: {embeddings}")

        if models:
            print("Embedding models:")
            for model in models:
                print(f"  - {model['model']}")
        else:
            print("Embedding models: None")

        if chunks > embeddings:
            print(
                "\nWarning: Some chunks do not have embeddings. "
                "Run /ingest again."
            )

    except sqlite3.Error as exc:
        print(f"RAG status error: {exc}")


def handle_evaluate(rag_query_service) -> None:
    """Run RAG retrieval evaluation and display the results."""

    print("\nStarting RAG evaluation...")
    print(f"Evaluation questions: {len(RAG_EVALUATION_CASES)}")
    print("Retrieval setting: Top 5")
    print("-" * 65)

    try:
        evaluator = RAGEvaluator(
            query_service=rag_query_service,
            k=5,
        )

        report = evaluator.evaluate(RAG_EVALUATION_CASES)

        print("\nRAG EVALUATION REPORT")
        print("=" * 65)
        print(f"Total questions       : {report.total_questions}")
        print(f"Precision@{report.k}           : {report.precision_at_k:.4f}")
        print(f"Precision@Retrieved : {report.precision_at_retrieved:.4f}")
        print(f"Recall@{report.k}              : {report.recall_at_k:.4f}")
        print(f"MRR                   : {report.mrr:.4f}")
        print(f"Hit Rate@{report.k}            : {report.hit_rate_at_k:.4f}")

        print("\nPER-QUESTION RESULTS")
        print("-" * 65)

        for index, result in enumerate(report.case_results, start=1):
            print(f"\n{index}. {result.question}")
            print(f"   Precision@{report.k}: {result.precision_at_k:.4f}")
            print(f"   Precision@Retrieved: {result.precision_at_retrieved:.4f}")
            print(f"   Recall@{report.k}: {result.recall_at_k:.4f}")
            print(f"   Reciprocal Rank: {result.reciprocal_rank:.4f}")
            print(f"   Hit Rate: {result.hit_rate_at_k:.4f}")

            print("   Retrieved sources:")
            if result.retrieved_sources:
                for source in result.retrieved_sources:
                    print(f"      - {source}")
            else:
                print("      No sources retrieved")

        print("\n" + "=" * 65)
        print("RAG evaluation completed.")

    except Exception as exc:
        print(
            f"\nRAG evaluation failed: "
            f"{type(exc).__name__}: {exc}"
        )

    """Run RAG retrieval evaluation and display the results."""

    print("\nStarting RAG evaluation...")
    print(f"Evaluation questions: {len(RAG_EVALUATION_CASES)}")
    print("Retrieval setting: Top 5")
    print("-" * 65)

    try:
        evaluator = RAGEvaluator(
            query_service=rag_query_service,
            k=5,
        )

        report = evaluator.evaluate(RAG_EVALUATION_CASES)

        print("\nRAG EVALUATION REPORT")
        print("=" * 65)

        print(f"Total questions       : {report.total_questions}")
        print(f"Precision@{report.k}           : {report.precision_at_k:.4f}")
        print(
            f"Precision@Retrieved : "
            f"{report.precision_at_retrieved:.4f}"
        )
        print(f"Recall@{report.k}              : {report.recall_at_k:.4f}")
        print(f"MRR                   : {report.mrr:.4f}")
        print(f"Hit Rate@{report.k}            : {report.hit_rate_at_k:.4f}")

        print("\nPER-QUESTION RESULTS")
        print("-" * 65)

        for index, result in enumerate(report.case_results, start=1):
            print(f"\n{index}. {result.question}")
            print(f"   Precision@{report.k}: {result.precision_at_k:.4f}")
            print(
                f"   Precision@Retrieved: "
                f"{result.precision_at_retrieved:.4f}"
            )
            print(f"   Recall@{report.k}: {result.recall_at_k:.4f}")
            print(f"   Reciprocal Rank: {result.reciprocal_rank:.4f}")
            print(f"   Hit Rate: {result.hit_rate_at_k:.4f}")

            print("   Retrieved sources:")

            if result.retrieved_sources:
                for source in result.retrieved_sources:
                    print(f"      - {source}")
            else:
                print("      No sources retrieved")

        print("\n" + "=" * 65)
        print("RAG evaluation completed.")

    except Exception as exc:
        print(
            f"\nRAG evaluation failed: "
            f"{type(exc).__name__}: {exc}"
        )

    """Run RAG retrieval evaluation and display the results."""

    print("\nStarting RAG evaluation...")
    print(f"Evaluation questions: {len(RAG_EVALUATION_CASES)}")
    print("Retrieval setting: Top 5")
    print("-" * 65)

    try:
        evaluator = RAGEvaluator(
            query_service=rag_query_service,
            k=5,
        )

        report = evaluator.evaluate(RAG_EVALUATION_CASES)

        print("\nRAG EVALUATION REPORT")
        print("=" * 65)

        print(f"Total questions       : {report.total_questions}")
        print(f"Precision@{report.k}           : {report.precision_at_k:.4f}")
        print(
            f"Precision@Retrieved : "
            f"{report.precision_at_retrieved:.4f}"
        )
        print(f"Recall@{report.k}              : {report.recall_at_k:.4f}")
        print(f"MRR                   : {report.mrr:.4f}")
        print(f"Hit Rate@{report.k}            : {report.hit_rate_at_k:.4f}")

        print("\nPER-QUESTION RESULTS")
        print("-" * 65)

        for index, result in enumerate(report.case_results, start=1):
            print(f"\n{index}. {result.question}")

            print(
                f"   Precision@{report.k}: "
                f"{result.precision_at_k:.4f}"
            )
            print(
                f"   Precision@Retrieved: "
                f"{result.precision_at_retrieved:.4f}"
            )
            print(
                f"   Recall@{report.k}: "
                f"{result.recall_at_k:.4f}"
            )
            print(
                f"   Reciprocal Rank: "
                f"{result.reciprocal_rank:.4f}"
            )
            print(
                f"   Hit Rate: "
                f"{result.hit_rate_at_k:.4f}"
            )

            print("   Retrieved sources:")

            if result.retrieved_sources:
                for source in result.retrieved_sources:
                    print(f"      - {source}")
            else:
                print("      No sources retrieved")

        print("\n" + "=" * 65)
        print("RAG evaluation completed.")

    except Exception as exc:
        print(f"\nRAG evaluation failed: {type(exc).__name__}: {exc}")

    """Run RAG retrieval evaluation and display the results."""

    print("\nStarting RAG evaluation...")
    print(f"Evaluation questions: {len(RAG_EVALUATION_CASES)}")
    print("Retrieval setting: Top 5")
    print("-" * 65)

    try:
        evaluator = RAGEvaluator(
            query_service=rag_query_service,
            k=5,
        )

        report = evaluator.evaluate(RAG_EVALUATION_CASES)

        print("\nRAG EVALUATION REPORT")
        print("=" * 65)

        print(f"Total questions : {report.total_questions}")
        print(f"Precision@{report.k}     : {report.precision_at_k:.4f}")
        print(f"Recall@{report.k}        : {report.recall_at_k:.4f}")
        print(f"MRR             : {report.mrr:.4f}")
        print(f"Hit Rate@{report.k}      : {report.hit_rate_at_k:.4f}")

        print("\nPER-QUESTION RESULTS")
        print("-" * 65)

        for index, result in enumerate(report.case_results, start=1):
            print(f"\n{index}. {result.question}")
            print(f"   Precision@{report.k}: {result.precision_at_k:.4f}")
            print(f"   Recall@{report.k}   : {result.recall_at_k:.4f}")
            print(f"   Reciprocal Rank    : {result.reciprocal_rank:.4f}")
            print(f"   Hit Rate           : {result.hit_rate_at_k:.4f}")

            print("   Retrieved sources:")

            if result.retrieved_sources:
                for source in result.retrieved_sources:
                    print(f"      - {source}")
            else:
                print("      No sources retrieved")

        print("\n" + "=" * 65)
        print("RAG evaluation completed.")

    except Exception as exc:
        print(f"\nRAG evaluation failed: {exc}")

def main() -> None:
    # Initialize the application's SQLite database.
    initialize_database()

    # Initialize the RAG tables in the same database.
    rag_store = RAGStore()
    rag_store.initialize_schema()

    ingestion_service = RAGIngestionService(store=rag_store)

    # RAG query embeddings are initialized lazily.
    rag_query_service = LazyRAGQueryService(rag_store)

    service = LLMService(
        rag_query_service=rag_query_service,
    )

    active_mode = "general"

    try:
        market_data = MarketDataTool(
            provider=AlphaVantageProvider()
        )
    except RuntimeError as exc:
        print(f"International market data unavailable: {exc}")
        market_data = MarketDataTool()

    conversation_id = service.start_conversation(
        title="Commodity Finance Chat"
    )

    print("\nCommodity Finance Assistance")
    print("Ask about commodities or investment concepts.")
    print(
    "Commands: /history, /conversations, /resume <id>, "
    "/quote <commodity>, /mcx-import [csv_path], "
    "/mcx <commodity>, /ingest [directory], "
    "/rag-status, /evaluate, "
    "/mode [general|trader|investor], /new, /exit,/approvals, /approve <request_id> <approver_id> [reason],/reject <request_id> <approver_id> <reason>"
    )
    print(f"Active mode: {active_mode.upper()}")
    print(f"Active conversation ID: {conversation_id}")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command in {"/exit", "/quit"}:
                print("Goodbye.")
                break

            if command == "/history":
                print_history(service, conversation_id)
                continue

            if command == "/conversations":
                print_conversations(service)
                continue

            if command == "/new":
                conversation_id = service.start_conversation(
                    title="Commodity Finance Chat"
                )
                print(
                    f"Started a new conversation: {conversation_id}"
                )
                continue

            if command == "/resume" or command.startswith("/resume "):
                parts = user_input.split(maxsplit=1)

                if len(parts) != 2 or not parts[1].strip():
                    print("Usage: /resume <conversation_id>")
                    continue

                requested_id = parts[1].strip()
                conversation = service.memory.get_conversation(
                    requested_id
                )

                if conversation is None:
                    print(
                        "Conversation not found. "
                        "Use /conversations to list saved chats."
                    )
                    continue

                conversation_id = requested_id
                title = conversation["title"] or "Untitled conversation"

                print(f"Resumed: {title}")
                print(f"Active conversation ID: {conversation_id}")
                print_history(service, conversation_id)
                continue

            if command == "/quote" or command.startswith("/quote "):
                handle_quote(market_data, user_input)
                continue

            if command == "/mcx-import" or command.startswith("/mcx-import "):
                handle_mcx_import(user_input)
                continue

            if command == "/mcx" or command.startswith("/mcx "):
                handle_mcx(user_input)
                continue

            if command == "/ingest" or command.startswith("/ingest "):
                handle_ingest(
                    user_input,
                    ingestion_service,
                    rag_store,
                )
                continue

            if command == "/rag-status":
                handle_rag_status(rag_store)
                continue
            if command == "/evaluate":
                handle_evaluate(rag_query_service)
                continue

            if command == "/mode":
                print(f"Current mode: {active_mode}")
                print("Available modes: general, trader, investor")
                continue

            if command.startswith("/mode "):
                requested_mode = (
                    user_input.split(maxsplit=1)[1].strip().lower()
                )
            if command == "/approvals":
                handle_approvals()
                continue
            if command == "/approve" or command.startswith("/approve "):
                parts = user_input.split(maxsplit=3)
                if len(parts) < 3:
                    print("Usage: /approve <request_id> <approver_id> [reason]")
                    continue

                reason = parts[3] if len(parts) == 4 else "Approved via local CLI."
                handle_approve(parts[1], parts[2], reason)
                continue
            if command == "/reject" or command.startswith("/reject "):
                parts = user_input.split(maxsplit=3)
                if len(parts) < 4:
                    print("Usage: /reject <request_id> <approver_id> <reason>")
                    continue

                handle_reject(parts[1], parts[2], parts[3])
                continue

                try:
                    get_mode_instruction(requested_mode)
                except ValueError as exc:
                    print(f"Mode error: {exc}")
                    continue

                active_mode = requested_mode
                print(
                    f"Assistant mode changed to: "
                    f"{active_mode.upper()}"
                )
                continue

            answer = service.reply(
                conversation_id,
                user_input,
                mode_instruction=get_mode_instruction(active_mode),
            )

            print(f"\nAssistant: {answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        except Exception as exc:
            print(
                f"\nError: {type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
