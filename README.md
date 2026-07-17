# EasySupport-AI

Um assistente inteligente de suporte corporativo desenvolvido em Python, combinando:

- Classificação de urgência com modelo de machine learning tradicional
- Recuperação Augmentada por Geração (RAG) usando Chroma e embeddings OpenAI
- Geração de respostas com OpenAI Responses
- Governança de ações com MCP (tooling e approval workflows)
- Interface interativa em Streamlit

## Visão geral

O projeto foi criado para ser um sistema compacto, testável e rastreável, com foco em segurança e controle humano.

Fluxo principal:

1. O usuário envia uma pergunta via Streamlit
2. O texto é classificado quanto à urgência
3. Documentos de conhecimento são recuperados e o contexto é construído
4. O modelo de linguagem gera uma resposta fundamentada nas fontes
5. Chamados de urgência alta podem ser propostos e exigem aprovação antes de serem abertos

## Estrutura do repositório

- `app.py` — front-end em Streamlit
- `config.py` — carregamento de variáveis de ambiente e configurações
- `workflow/graph.py` — grafo de estado que controla a validação, classificação, recuperação, decisão e geração
- `classifier/` — treinamento e serviço de classificação de urgência
- `rag/` — indexação e busca de conhecimento com Chroma
- `generation/` — serviço de geração de respostas com OpenAI
- `mcp_server/` — ferramentas MCP para criação e consulta de tickets
- `knowledge_base/` — documentos de referência usados para RAG
- `data/` — dados de treinamento, índices e artefatos gerados
- `tests/` — testes automatizados

## Requisitos

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração

Copie o arquivo de exemplo e configure as variáveis de ambiente:

```bash
copy .env.example .env
```

Edite o `.env` e defina pelo menos:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (opcional)
- `OPENAI_EMBEDDING_MODEL` (opcional)
- `RAG_TOP_K` (opcional)
- `RAG_MIN_SCORE` (opcional)
- `MAX_CONTEXT_TOKENS` (opcional)
- `CLASSIFIER_CONFIDENCE_MIN` (opcional)

## Preparar o sistema

1. Treine ou valide o classificador de urgência:

```bash
python classifier/train.py
```

2. Gere o índice de documentos da base de conhecimento:

```bash
python rag/index_documents.py
```

## Execução

Inicie a interface:

```bash
streamlit run app.py
```

A aplicação usa o servidor MCP internamente para executar ferramentas de ticket via `mcp_server/server.py`.

## Comportamento principal

- O classificador prevê urgência e confiança com base no texto do chamado.
- O componente RAG recupera trechos de `knowledge_base/` usando embeddings.
- O assistente responde apenas com base em evidências recuperadas.
- Chamados de urgência alta são encaminhados para aprovação antes de abrir um ticket.

## Testes

Execute os testes com:

```bash
pytest
```

## Observabilidade e logs

O projeto já inclui suporte básico de logging e persistência de tickets em SQLite via `data/support.db`.

## Nota

Este repositório serve como um exemplo de integração entre ML clássico, RAG, MCP e uma interface conversacional. Ajuste os prompts, documentos e limites de confiança conforme os requisitos do seu ambiente.

---

# EasySupport-AI - English

An intelligent corporate support assistant built in Python, combining:

- urgency classification with a traditional machine learning model
- Retrieval-Augmented Generation (RAG) using Chroma and OpenAI embeddings
- answer generation with OpenAI Responses
- action governance with MCP (tooling and approval workflows)
- interactive interface in Streamlit

## Overview

The project is designed to be compact, testable, and traceable, with a focus on safety and human oversight.

Main flow:

1. The user submits a question through Streamlit
2. The text is classified for urgency
3. knowledge documents are retrieved and context is built
4. the language model generates an answer grounded in sources
5. high-urgency tickets may be proposed and require approval before being created

## Repository structure

- `app.py` — Streamlit front-end
- `config.py` — environment variable loading and configuration
- `workflow/graph.py` — state graph controlling validation, classification, retrieval, decision, and generation
- `classifier/` — urgency classifier training and service
- `rag/` — knowledge indexing and search with Chroma
- `generation/` — answer generation service using OpenAI
- `mcp_server/` — MCP tools for ticket creation and lookup
- `knowledge_base/` — reference documents used for RAG
- `data/` — training data, indexes, and generated artifacts
- `tests/` — automated tests

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Copy the example file and configure environment variables:

```bash
copy .env.example .env
```

Edit `.env` and set at least:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional)
- `OPENAI_EMBEDDING_MODEL` (optional)
- `RAG_TOP_K` (optional)
- `RAG_MIN_SCORE` (optional)
- `MAX_CONTEXT_TOKENS` (optional)
- `CLASSIFIER_CONFIDENCE_MIN` (optional)

## Setup

1. Train or validate the urgency classifier:

```bash
python classifier/train.py
```

2. Generate the knowledge base document index:

```bash
python rag/index_documents.py
```

## Running

Start the interface:

```bash
streamlit run app.py
```

The application uses an MCP server internally to execute ticket tools via `mcp_server/server.py`.

## Main behavior

- the classifier predicts urgency and confidence based on the ticket text
- the RAG component retrieves snippets from `knowledge_base/` using embeddings
- the assistant answers only based on retrieved evidence
- high-urgency tickets are routed for approval before opening a ticket

## Tests

Run tests with:

```bash
pytest
```

## Observability and logs

The project includes basic logging support and ticket persistence in SQLite via `data/support.db`.

## Note

This repository is an example of integrating classic ML, RAG, MCP, and a conversational interface. Adjust prompts, documents, and confidence thresholds to meet your environment requirements.