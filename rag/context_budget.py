import tiktoken

from config import load_settings


def get_encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def build_context(documents: list[dict]) -> tuple[str, list[dict], int]:
    settings = load_settings()
    encoding = get_encoding(settings.model_name)

    selected = []
    blocks = []
    used_tokens = 0

    for document in documents:
        block = (
            f"FONTE: {document['source']}\n"
            f"CHUNK: {document['chunk_id']}\n"
            f"CONTEÚDO:\n{document['content']}"
        )

        block_tokens = len(encoding.encode(block))

        if used_tokens + block_tokens > settings.max_context_tokens:
            break

        blocks.append(block)
        selected.append(document)
        used_tokens += block_tokens

    return "\n\n---\n\n".join(blocks), selected, used_tokens