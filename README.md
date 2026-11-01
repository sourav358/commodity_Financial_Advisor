Commodity Trader & Investor Finance Assistant
An Agentic AI learning project that helps commodity traders and investors explore commodity-market information, understand financial concepts, retrieve supporting knowledge from documents, and interact with market-data tools through a conversational CLI.
Project status: Active MVP development. The project includes RAG, MCP-based tools, trader/investor modes, guardrails, human-in-the-loop approvals, local conversation memory, and observability. Market data and example datasets may be synthetic or locally supplied; do not treat generated outputs as financial advice or live exchange prices unless explicitly verified against an authoritative source.

Key capabilities
- Conversational assistant: LLM-backed interaction through a command-line interface.
- Trader and investor modes: Switch between general, trader, and investor contexts.
- Retrieval-Augmented Generation (RAG): Load documents, split them into chunks, create embeddings, persist the index, and retrieve relevant context for answers.
- Atomic RAG ingestion: Persist a document, its chunks, and embeddings together so failed re-ingestion does not leave a partially updated index.
- MCP integration: Expose commodity-related tools through an MCP server/client workflow.
- Skills and tool permissions: Select relevant skills and constrain which tools can be used.
- Guardrails: Validate user input, assistant output, and tool calls.
- Human-in-the-loop (HITL): Route approval-required actions through an explicit approval/rejection workflow with auditability.
- Conversation memory: Store and retrieve local conversation history using SQLite.
- Market-data utilities: Work with commodity quotes and locally imported MCX bhavcopy data.
- Observability: Structured application logging and operational visibility.
- Evaluation: Run a RAG evaluation set and inspect retrieval metrics.
Technology stack
- Python 3.12
- Poetry
- SQLite for local persistence and tests
- OpenAI API for supported LLM and embedding operations
- YAML-based model/application configuration
- MCP for tool integration
- pytest for automated tests
Architecture overview
User
  |
  v
CLI
  |
  v
LLM Service / Orchestration
  |------ Mode selection (general / trader / investor)
  |------ Skills registry and tool permissions
  |------ Input / output / tool guardrails
  |------ Conversation memory
  |------ Observability and logging
  |
  +------ RAG pipeline
  |         |-- Document loading and chunking
  |         |-- Embedding generation
  |         |-- SQLite document/chunk/embedding store
  |         |-- Retrieval and context construction
  |         `-- Evaluation
  |
  +------ MCP client / commodity tools
  |
  `------ HITL approval workflow
            |-- Approval request
            |-- Human approve / reject
            `-- Audit trail
Project structure
The package is maintained directly under commodity_finance_assistance/ (without a src/ directory). Major areas include:
commodity_finance_assistance/
├── agent/           # Agent orchestration and related logic
├── cache/           # Caching components
├── context/         # Context construction, indexers, and retrievers
├── domain/          # Commodity, portfolio, and risk domain logic
├── evaluations/     # Evaluation datasets and metrics
├── guardrails/      # Input, output, and tool validation
├── hitl/            # Human approval workflow
├── llm/             # LLM service and provider integration
├── mcp/             # MCP client/server integration
├── memory/          # SQLite-backed conversation memory
├── observability/   # Logging and instrumentation
├── rag/             # RAG ingestion, pipeline, and storage
├── skills/          # Skill registry and permissions
├── tasks/           # Task workflows
├── tools/           # Commodity and finance tools
└── cli.py           # Command-line entry point

tests/               # Automated tests
config.yml           # Application and model configuration
pyproject.toml       # Poetry project and dependencies
Some module names may evolve as the MVP is refined.
Prerequisites
- Windows, macOS, or Linux
- Python 3.12.x
- Poetry 2.x
- An OpenAI API key if using OpenAI-backed models
Check your versions:
python --version
poetry --version
Setup
1. Clone the repository and enter the project directory:
   git clone https://github.com/sourav358/commodity_Financial_Advisor.git
   cd commodity_Financial_Advisor
2. Configure Poetry to use Python 3.12 and install dependencies:
   poetry env use 3.12
   poetry install
3. Configure your API key as an environment variable. Do not commit API keys:
   PowerShell
   $env:OPENAI_API_KEY = "your_api_key_here"
   Command Prompt
   set OPENAI_API_KEY=your_api_key_here
4. Review config.yml and ensure the configured model provider, chat model, and embedding model match the credentials and dependencies available in your environment.
5. Start the CLI:
   poetry run python -m commodity_finance_assistance.cli
Keep secrets in environment variables or a local, ignored .env file. Never paste API keys into source code, configuration committed to Git, logs, or issue reports.

CLI commands
Command	Purpose
/mode	Show the active mode and available modes
/mode general	Switch to general assistant mode
/mode trader	Switch to trader mode
/mode investor	Switch to investor mode
/quote <commodity>	Request a commodity quote using the configured data/tool workflow
/mcx <commodity>	Query locally available MCX commodity data
/mcx-import [csv_path]	Import an MCX bhavcopy CSV
/ingest [directory]	Ingest documents from a directory into the RAG index
/rag-status	Display document, chunk, and embedding counts and embedding coverage
/evaluate	Run the configured RAG evaluation set
/history	View conversation history
/conversations	List available conversations
/resume <id>	Resume a conversation
/new	Start a new conversation
/approvals	List pending approval requests
/approve <request_id> <approver_id> [reason]	Approve a pending request
/reject <request_id> <approver_id> <reason>	Reject a pending request
/exit	Exit the CLI


The available command list printed by the running CLI is the source of truth if commands change between versions.
Testing and quality checks
Run the test suite:
poetry run pytest -q
Compile the package:
poetry run python -m compileall commodity_finance_assistance
Run the CLI smoke test:
poetry run python -m commodity_finance_assistance.cli
Then, from the CLI, verify at least:
/mode
/mode trader
/mode investor
/mode general
/rag-status
/evaluate
/approvals
/history
/exit
Latest reported test result
The latest test output supplied during development was:
262 passed, 2 subtests passed
The package also compiled successfully, and /rag-status reported 6 documents, 36 chunks, 36 embeddings, 0 chunks missing embeddings, and 100% embedding coverage. These are development-session results, not a guarantee that every environment or future commit will produce identical results.
RAG evaluation
The reported evaluation run used 20 questions and produced:
Metric	Result
Precision@5	0.7000
Precision@Retrieved	0.7000
Recall@5	1.0000
Mean Reciprocal Rank (MRR)	0.9350
Hit Rate@5	1.0000


The current improvement focus is retrieval precision while preserving recall. Re-run /evaluate after changing chunking, retrieval, ranking, or embedding behavior.
Data and financial-use disclaimer
This is an educational software project, not a trading system, investment adviser, broker, or exchange feed. Synthetic, imported, cached, or model-generated data may be incomplete, delayed, inaccurate, or stale. Verify prices, contract specifications, market rules, and calculations with authoritative sources before relying on them. Do not use the application to execute trades or make investment decisions without independent review.
Contributing / development workflow
1. Create a feature branch.
2. Make a focused change and add or update tests.
3. Run compilation and the complete test suite.
4. Review git diff and ensure no secrets, local databases, generated files, or private documents are included.
5. Commit with a descriptive message and push the branch.
License
No license has been specified yet. Add a LICENSE file before presenting the repository as open source or permitting reuse by others.