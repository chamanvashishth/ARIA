from aria.rag import (
    BM25Retriever,
    LocalRAGPipeline,
    TextChunker,
    TextDocument,
    tokenize_terms,
)


def test_chunker_preserves_source_metadata_and_overlap() -> None:
    chunker = TextChunker(chunk_size=10, overlap=3)
    document = TextDocument("notes/quantum.txt", "abcdefghijklmnop", {"kind": "notes"})
    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "hijklmnop"]
    assert chunks[0].source == "notes/quantum.txt"
    assert chunks[0].metadata == {"kind": "notes"}
    assert chunks[0].id == chunker.chunk(document)[0].id


def test_chunker_handles_empty_text_and_invalid_sizes() -> None:
    assert TextChunker().chunk(TextDocument("empty.txt", "")) == []
    for args in ({"chunk_size": 0}, {"chunk_size": 10, "overlap": 10}):
        try:
            TextChunker(**args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunker settings should fail")


def test_bm25_retrieves_lexical_matches_and_respects_top_k() -> None:
    pipeline = LocalRAGPipeline(chunker=TextChunker(chunk_size=200, overlap=0))
    pipeline.add_documents(
        [
            TextDocument("quantum.txt", "Quantum computing uses qubits and quantum gates."),
            TextDocument("garden.txt", "Tomatoes grow well in sunny gardens."),
            TextDocument("physics.txt", "Entanglement links the states of distant qubits."),
        ]
    )

    results = pipeline.retrieve("qubits quantum", top_k=2)
    assert len(results) == 2
    assert results[0][0].source == "quantum.txt"
    assert results[0][1] > results[1][1]
    assert pipeline.retrieve("nonmatchingterm") == []


def test_context_includes_provenance_and_obeys_character_cap() -> None:
    pipeline = LocalRAGPipeline(
        chunker=TextChunker(chunk_size=100, overlap=0), top_k=2
    )
    pipeline.add_document(
        TextDocument("guide.md", "A qubit can exist in a superposition of basis states.")
    )

    context = pipeline.build_context("qubit superposition", max_characters=80)
    assert "[Source: guide.md | Chunk: 0]" in context
    assert len(context) <= 80


def test_bm25_empty_query_and_top_k_validation() -> None:
    retriever = BM25Retriever()
    assert retriever.search("anything") == []
    assert tokenize_terms("Qubit, qubit! हिन्दी") == ["qubit", "qubit", "हिन्दी"]
    try:
        retriever.search("query", top_k=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative top_k should fail")
