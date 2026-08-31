# Engineering deep dive

The full technique-by-technique breakdown for [RAG Engine](README.md) — how
each piece actually works, what real data verified it, every real bug found
building it, the evaluation results, and the performance/latency work. The
top-level README covers what the project is and how to run it; this is
where the depth lives.

## Structured data

CSV, Excel (`.xlsx`/`.xls`), and JSON/`.jsonl` files are picked up automatically
by `python cli.py ingest` from `data/raw/`. Each row/record becomes one
Document: its fields are flattened into a `field: value` text block for
embedding, while the original typed values are kept in `metadata` (e.g.
`row_index`, `sheet` for Excel). Excel files with multiple sheets are fully
supported; every sheet is loaded and tagged with its sheet name.

You can also pull rows from an existing SQL table into the vector store the
same way:

```bash
python cli.py ingest --sql-conn "sqlite:///path/to.db" --sql-table some_table
```

Any SQLAlchemy-compatible connection string works (Postgres, MySQL, etc.) as
long as the matching DBAPI driver is installed (e.g. `psycopg2-binary` for
Postgres) — SQLite's driver ships with Python. Use `--sql-query` instead of
`--sql-table` for a custom `SELECT` rather than a full-table dump.

**Important caveat**: this row-to-text pattern is for *lookup* questions
("what's X's plan?", "what caused incident Y?") — vector search finds
matching rows, it doesn't compute. It cannot answer aggregate questions
("what's the total?", "which one has the most sections?") correctly, because
no single retrieved chunk contains the answer. That's what Text-to-SQL below
is for.

## Web search

Set `TAVILY_API_KEY` in `.env` (free tier: 1,000 searches/month at
[tavily.com](https://tavily.com)) and web search becomes available as a
fallback source — see [Corrective RAG](#corrective-rag-c-rag) for when it
actually gets used. Unset the key, or set `WEB_SEARCH_ENABLED=false`, to turn
it off entirely (the system just answers from local docs alone, or says it
doesn't know). Answers cite web sources by URL so it's clear when a claim
came from the internet rather than your own documents.

## Text-to-SQL

The row-to-text pattern above is deliberately limited to lookups. For
structured data you actually want to *compute over* — counts, sums, "which X
has the most Y" — there's a second, independent path that keeps the data
genuinely relational instead of flattening it to text:

```bash
python cli.py build-sql-db          # builds data/structured.db from every csv/xlsx in data/raw/
python cli.py sql "which subject has the most course sections?"
```

`build-sql-db` (`src/ingestion/sql_builder.py`) loads every CSV/Excel file
under `data/raw/` into its own SQLite table (one table per sheet for Excel;
column names sanitized to valid SQL identifiers; fully-blank rows dropped).
`cli.py sql` then runs the actual text-to-SQL chain
(`src/rag/text_to_sql.py`): the LLM sees the schema, writes a SQL query,
which gets executed for real, and a final LLM call turns the result into a
plain-language answer.

Because an LLM-generated query executing against a real database is a real
risk surface — a cleverly worded question could try to coerce a
`DROP`/`DELETE`, the same class of concern as SQL injection even though the
"attacker" here is the model rather than raw user input — there are two
independent guards, so one being wrong isn't enough to matter:

1. The database connection itself is opened **read-only** at the SQLite level
   (`file:...?mode=ro`), so a write physically cannot succeed regardless of
   what SQL is generated.
2. Every generated query is validated as a single `SELECT`/`WITH` statement —
   rejecting semicolon-stacked statements and any `INSERT`/`UPDATE`/`DELETE`/
   `DROP`/`ALTER`/`ATTACH`/`PRAGMA`/etc. — before it's ever executed.

**Two real bugs found running this live, not in a unit test**: (1) the code
that strips a markdown fence off the model's SQL output only matched the
literal ` ```sql ` tag — when the model fenced a query with ` ```sqlite `
instead (a real, common variant), the regex partial-matched and left `ite\n`
prepended to the query, e.g. `'ite\nSELECT AVG("cap_enrl") FROM ...'`. The
safety guard correctly rejected it (it doesn't start with `SELECT`/`WITH`) —
that part worked exactly as designed. But (2) that rejection raised an
exception nothing caught, which propagated out of `ChatSession.ask()` and
crashed the *entire session*, not just that one answer. Both are fixed: the
fence-stripping regex now matches any language tag (`^```\w*\s*`, not just
`sql` literally), and `pipeline.py` catches any exception from the SQL path
and falls back to the RAG path for that same question instead of crashing —
consistent with how every other component in this project degrades (skip,
don't crash) rather than propagating a failure into a hard stop.

## Routing

With two independent ways to answer a question now — RAG over documents, or
text-to-SQL over real tables — something has to decide which one a given
question actually needs. `src/rag/router.py` classifies each question with a
structured-output LLM call, given the real SQL schema (`get_table_info()`) so
the decision is grounded in what's actually queryable, not a guess. If no SQL
database has been built yet, the router **skips the LLM call entirely** and
routes to the vectorstore by default — there's nothing to route to otherwise,
and there's no reason to spend a call finding that out.

This is what makes `cli.py chat` and the `/query` endpoint a single interface
instead of three separate tools the user has to pick between: ask anything,
and the system figures out whether it's a lookup or a computation. Both
interfaces return which route handled the question (`route: "sql"` or
`"vectorstore"`) for transparency.

## Query analysis

Two independent transformations of the question happen before retrieval,
both in `src/rag/query_analysis.py`:

- **Decomposition**: a compound question — "what's the drop deadline, and
  what's the mission statement?" — doesn't retrieve well as one query: a
  single embedding/BM25/rerank pass tends to be dominated by whichever half's
  wording is more prominent, so the other half silently goes unanswered.
  `decompose()` splits it into atomic parts with a structured-output LLM
  call; atomic questions get a one-element list back, so nothing extra
  happens for the common case.
- **Step-back prompting**: a narrow question ("what's the deadline to drop
  ACCT 110 with a W?") might use different wording than the source document,
  or need surrounding policy context to answer well. `step_back()` generates
  ONE broader/more general version of the question (e.g. "what are the
  course withdrawal deadlines and policies?"), which gets retrieved
  alongside the original — unless it comes back unchanged (the question was
  already general), in which case it's skipped rather than wastefully
  retrieved twice.

Both are gated by `needs_decomposition()` — a cheap upfront check: for an
already-simple, self-contained question, decompose+step-back would just add
two LLM calls and double the downstream retrieval/grading fan-out for no
benefit. That decision isn't made here anymore, though — it's decided once,
upstream, by [the scope gate](#scope-gate), combined into the very same LLM
call as the scope check (they're two independent yes/no questions about the
same input). `DecomposingRetriever` itself now always decomposes when
invoked; `needs_decomposition()` still lives in this file since it's a
query-analysis concern, and the scope gate calls into it directly only for
the rarer case where there's no catalog/web fallback to combine the check
with. See [Performance](#performance) for how this gate's first version
barely worked, and the real fix.

Every resulting (sub-/step-back) question is retrieved independently through
the same reranking retriever, and the results are merged + deduplicated.

**A real bug this surfaced, worth knowing about**: the first version reranked
the pooled candidates once against the *original* combined question. That
technically worked but was subtly wrong — testing it against the compound
question above, the cross-encoder scored calendar-date chunks as more
relevant to the full question text than the mission-statement chunk (which
was genuinely retrieved, just outscored), so the final top-N context dropped
it and the model correctly said it had no answer for that half. The fix:
rerank **per sub-question**, before merging — `retrieval.py`'s
`build_hybrid_retriever` wraps the ensemble in the reranker *first*, then
wraps *that* in the decomposing retriever, so each sub-question gets reranked
against its own wording, not the combined one. Confirmed against the same
real question afterward: both halves answered correctly, each citing the
right source document.

## Self-query retrieval

Instead of always embedding the raw question as-is, `src/rag/self_query.py`
uses an LLM to split it into (a) the actual semantic search text and (b)
structured metadata filters — auto-generated from the question, not written
by hand. Asking "what does the academic calendar say about November
deadlines?" becomes a semantic search for "November deadlines" filtered to
`source == "RITK Academic Calendar 2026-27 Final.pdf"`, a filter the question
never stated explicitly.

Only `source` and `sheet` (which sheet, for Excel-derived rows) are declared
as filterable attributes, and both are grounded in the real distinct values
actually present in the store (`build_self_query_retriever` reads them
straight out of Chroma) rather than left for the LLM to guess at plausible
filenames. Arbitrary CSV/JSON columns aren't filterable this way — their
names and meaning vary per ingested file and can't be described generically
up front; that's a job for text-to-SQL instead, over data that's genuinely
relational rather than flattened to text.

Each source's `AttributeInfo` description also includes its one-sentence
entry from the [source catalog](#source-catalog), not just its bare
filename/URL — so "the mission statement" or "the scholarship policy" can
resolve to the right file even when the question never says the filename,
because the LLM can see what each source is actually *about*.

This joins the vector and BM25 retrievers as a third local candidate source
in the same reranked ensemble (`retrieval.py`'s `_build_local_ensemble`) —
`None` (skipped) if nothing's been ingested yet, since there's no metadata to
build filterable attributes from. It's also now *conditional per question* —
`ConditionalSelfQueryRetriever` skips the LLM-based filter-construction call
entirely unless the question shares real vocabulary with a known source's
filename or catalog description (a free, non-LLM keyword-overlap check), so
the expensive part doesn't run on every question regardless of whether
metadata filtering could plausibly help. Same conservative fail-open default
as the scope gate: any overlap at all still runs it.

**Verified against real data**: asking about "the academic calendar" and
"November deadlines" returned candidates filtered to *only*
`RITK Academic Calendar 2026-27 Final.pdf` — none of the handbook or course
spreadsheet chunks leaked in, even though nothing in the question stated the
filename explicitly.

Because the query-construction LLM call can occasionally produce a malformed
filter for an edge-case question, the retriever is wrapped in a
`SafeRetriever` that swallows exceptions and returns no results for that
attempt — self-query failing should degrade to "the other retrievers still
ran," not break the whole ensemble.

## Source catalog

`src/rag/source_catalog.py` generates a one-sentence, LLM-written
description for every distinct source in the store — one call per source,
the first time it's seen. Descriptions are persisted to `SOURCE_CATALOG_PATH`
(a JSON file) and reused on every later call, so re-opening a `ChatSession`
never re-summarizes content that hasn't changed; only genuinely new sources
cost anything. Entries for sources no longer present in the store are
dropped on the next build, so the file can't drift into staleness either.

This exists to serve two other components — it isn't useful on its own:
[self-query retrieval](#self-query-retrieval) (richer filter descriptions)
and the [scope gate](#scope-gate) below (a topic-relevance check). Both need
"what does each source roughly cover", and generating that once and sharing
it is simpler and cheaper than each component inventing its own version.

**Verified against real data**: run against 19 real ingested sources (4 PDFs,
1 Excel file, 14 web pages), every description was accurate and specific at
a glance — e.g. `SG Constitution.pdf` → "The source details the RITK Student
Government's constitution, covering its establishment, name, and logo,"
`WithdrawalForScholarships.pdf` → "This source explains how withdrawing from
courses affects RIT Kosovo (A.U.K) scholarships" — genuinely distinguishing
one source from another, not generic restatements of the filename.

## Scope gate

Corrective RAG (below) already falls back to web search when local content
turns out not to be useful — but that decision only happens *after* a full
retrieval pass: vector search, BM25, self-query, parent-document retrieval,
and reranking, all for a question that might have nothing to do with
anything ingested. `src/rag/scope_gate.py` adds a cheap check in front of
all of that: given the [source catalog](#source-catalog), does this question
plausibly relate to *any* known source at all? If clearly not, skip straight
to web search. If yes — or if there's nothing to check against yet — proceed
with local retrieval exactly as before.

It runs **once, against the original question, before decomposition** —
`ScopeGatedRetriever` dispatches to one of three retrievers (web-only,
simple, or decomposing — see below), rather than being wrapped one level
inside decomposition. Scoping a compound question as a whole is safe here
(unlike reranking/correction, which genuinely need to run per sub-question —
see [Query analysis](#query-analysis)): the gate only needs "does ANY of
this relate to something we have," which a single pass answers correctly.
For a clearly out-of-scope question this also means decomposition never
runs at all, since there's nothing local left to decompose for.

**The scope check and the decomposition check are the same LLM call, not
two.** `check_scope_and_decomposition()` asks both independent yes/no
questions (in scope? worth decomposing?) about the same input in one
structured-output round trip — the same fix already applied once to the two
self-correction checks (see [Self-correcting generation](#self-correcting-generation)),
applied here to save a second sequential round trip on every in-scope
question. `ScopeGatedRetriever` then dispatches on both answers at once:
out of scope → `web_retriever` directly; in scope and simple →
`simple_retriever` (no decompose/step-back); in scope and complex →
`complex_retriever` (`DecomposingRetriever`, which — now that the decision
was already made upstream — always decomposes when invoked, rather than
checking for itself). The combined call is skipped entirely (falling back
to asking `needs_decomposition()` alone) when there's nothing to gate
against (empty catalog) or nowhere to fall back to (no web retriever
configured) — in both cases a decomposition decision is still needed, just
not combined with a scope check that wouldn't mean anything.

This is a **latency/cost optimization layered on top of Corrective RAG, not
a replacement for it**. The two catch different failure modes: the scope
gate catches "this topic isn't covered by anything ingested, don't bother
retrieving at all"; Corrective RAG catches "this looked like it might be
covered, but the actual retrieved content wasn't good enough" — a case the
scope gate can't see coming, since it never looks at real content, only
short descriptions.

Deliberately biased toward false positives on both axes, not false
negatives: the prompt explicitly says to default to "in scope" and "needs
decomposition" whenever unsure, and an empty catalog (nothing ingested yet)
always passes scope through unchecked. A wrongly "in scope" question just
costs one extra retrieval-and-grade pass that Corrective RAG already
handles correctly; a wrongly "out of scope" question would skip local
content that actually had the answer — a real correctness regression, not
just wasted cost. Same shape for decomposition: wrongly skipping it for a
question that needed it is a real answer-quality regression, wrongly
running it just costs latency. Cheap mistakes are fine here; the other kind
isn't.

**Verified against real data**: asking about the Student Government
constitution correctly classified as in-scope. Asking for a lasagna recipe
and asking who won the most recent Formula 1 championship both correctly
classified as out-of-scope — and, run through the full pipeline, the lasagna
question returned an answer sourced *entirely* from real recipe websites
(zero RIT Kosovo sources cited), confirming local retrieval was skipped
entirely rather than run and simply come up empty. Separately, profiling a
simple in-scope question after the merge shows exactly one
`check_scope_and_decomposition` call and zero fallback-path
`needs_decomposition` calls (4 tracked calls total, down from 5) — confirmed
the merge is actually happening, not just theoretically wired up — and a
compound question profiled the same way correctly shows the combined call
firing once, followed by real `decompose`/`step_back`/multi-question
`grade_relevance` calls, proving the complex-question dispatch path still
works end to end. See [Performance](#performance) for the numbers.

## Parent-document retrieval

There's a real tension in choosing a chunk size: small chunks embed
precisely (a search for one specific fact matches cleanly, not diluted by
unrelated surrounding text), but a multi-step procedure or a paragraph that
spans a chunk boundary gets returned as an incomplete fragment. Parent-
document retrieval (`src/rag/parent_document.py`) resolves this by
indexing differently from what it returns: small **child** chunks
(`CHILD_CHUNK_SIZE`, default 400 chars) get embedded and matched for
precision, but the larger **parent** chunk they belong to (`PARENT_CHUNK_SIZE`,
default 2000 chars) is what actually comes back — LangChain's
`ParentDocumentRetriever` handles the child→parent lookup.

Kept as its own Chroma collection plus a persisted docstore
(`PARENT_DOCSTORE_DIR`) for the parent chunks, rather than sharing the main
collection: child chunks carry a `doc_id` pointing into that docstore, a
different metadata shape than every other retriever in this project assumes
(`source`/`sheet`/etc.), and mixing the two would make those assumptions
unreliable. It's also scoped to prose documents only (PDF/DOCX/TXT/MD) —
structured rows (CSV/Excel/JSON) are already atomic; there's no larger
"parent" a single row would benefit from being reunited with. `cli.py ingest`
feeds prose files into it automatically alongside the main collection.

**Verified against real data**: asking "what should a student do if they
disagree with a grade?" returned two context chunks of 1953 and 939
characters — well beyond the main collection's ~1000-char chunks — and the
answer walked through the full nested grade-appeal procedure (instructor →
written appeal to the Dean → Grade Appeal Committee → two possible outcomes)
coherently, in one piece, rather than as a fragment missing one of the
steps.

**A real bug this surfaced, worth knowing about**: adding a second Chroma
collection sharing the same persist directory as the main one caused an
intermittent `chromadb.errors.InternalError: ... Nothing found on disk` —
each `Chroma(...)` call was opening its own independent client against that
directory, and concurrent/rapid access from separate client objects to the
same on-disk store isn't safe. Fixed by sharing one cached
`chromadb.PersistentClient` (`vectorstore.get_chroma_client()`) across every
collection in the project, rather than letting each collection manage its
own client. (This same client singleton later turned out to have a second,
subtler concurrency bug of its own — see
[Web UI internals](#web-ui-internals).)

## Corrective RAG (C-RAG)

Earlier versions of this project blended live web search into every single
retrieval as a third ensemble member alongside vector and BM25 search. That
worked, but it's not actually Corrective RAG — it's just "more sources,
always." Real C-RAG uses a *quality signal* to decide whether to fall back at
all: `src/rag/corrective.py` grades the locally-retrieved documents for
relevance with a structured-output LLM call (strict — tangentially related
doesn't count), and only if too few pass (`CRAG_MIN_RELEVANT_DOCS`, default 1)
does it fall back to a live web search for that same (sub-)question. If no
documents were retrieved at all, grading is skipped entirely — there's
nothing to grade — same "don't call the LLM for a decision that's already
obvious" pattern as the router.

This is genuinely per-sub-question, not a one-time decision for the whole
compound question: `retrieval.py`'s `build_hybrid_retriever` wraps the graded/
corrective retriever *inside* `DecomposingRetriever`, so each sub-question
independently decides whether it needs the web, rather than one global
correction call for the merged pool.

**Verified against real data, not just the local wording**: asking "what is
RIT Kosovo's mission statement about?" answers entirely from the handbook —
every inline citation in the generated answer is the handbook, zero web
citations — because local retrieval graded sufficient for that question.
Asking "what is the current population of Prishtina, Kosovo?" (nothing in
any ingested document) answers entirely from web sources with real, current
figures, each correctly cited by URL — local retrieval graded insufficient,
correction kicked in, and the model didn't hallucinate a number or refuse to
answer.

**A precision gap that used to exist here, now fixed**: the `sources` field
returned alongside the answer used to be computed from *every* document
retrieved across all sub-questions, not just the ones the answer actually
cited. Because correction runs per (sub-/step-back) question independently,
the mission statement example above used to still list a few generic "what
is a mission statement" web pages in `sources` — the auto-generated
**step-back** question ("what is a mission statement in general?") doesn't
get answered by the handbook either, so *that* sub-question's own
correction pass pulled in web results, even though the original question
didn't need them. The model correctly ignored those when writing the answer
(confirmed: zero inline citations to them), but `sources` didn't
distinguish "retrieved" from "actually used."

Fixed by `pipeline.py`'s `_cited_sources()`: parses the literal
`[source: ...]` tags the model is required to copy exactly (see
`chain.py`'s `DOCUMENT_PROMPT`/`ANSWER_PROMPT`) out of the generated answer
text, intersected against the real retrieved sources as a guard against a
malformed or hallucinated tag. Re-ran the exact mission-statement question
that originally surfaced this: `sources` now returns exactly
`['Student Handbook (Code of Conduct)_RIT Kosovo.pdf']` — none of the
generic step-back web pages, matching what the answer actually cites.
Covered by pure-logic tests in
[`test_pipeline_helpers.py`](tests/test_pipeline_helpers.py) (deliberately
kept out of `test_pipeline.py`, which is module-skipped without
`GOOGLE_API_KEY` — this needed no credentials and should always run,
including in CI). Deliberately does NOT fall back to "all retrieved
sources" when nothing parses (e.g. an honest "the context doesn't cover
this" answer legitimately cites nothing) — that fallback would just
reintroduce the same bug for that case; an empty `sources` list is the
correct, honest answer there.

## Self-correcting generation

Corrective RAG grades *retrieval* quality, before generation ever happens.
It structurally cannot catch a different class of failure: retrieval looked
fine, but the model still wrote something ungrounded, or wrote something
that technically responds to the words in the question without actually
resolving it. That's what `src/rag/self_correction.py` grades instead — the
generated answer itself — mirroring the canonical LangGraph Self-RAG/RRR
("Rewrite-Retrieve-Read") flow: retrieve → generate → grade the answer →
correct, rather than stopping at "grade what was retrieved."

Two independent checks, both from a single combined LLM call
(`grade_answer`), run in `pipeline.py`'s `ChatSession._ask_vectorstore`
after every generation:

1. **Hallucination check** — is every claim in the answer actually traceable
   to the retrieved context? If not, the answer is **regenerated from the
   SAME context**, with explicit feedback about the failure ("your previous
   answer included claims not supported by the context — regenerate using
   ONLY facts explicitly stated") — not re-retrieved. Retrieval here is
   deterministic; re-running it with the same question would return the
   same context and likely the same hallucinated answer, so the fix has to
   happen at generation time, which is why `build_conversational_rag_chain`
   now exposes the underlying `document_chain` separately, so it can be
   invoked directly with the already-retrieved context.
2. **Answers-the-question check** — does the answer actually engage with
   what was asked (an honest "the context doesn't cover this" counts as
   answering)? If not, the question is **rewritten** and the **full
   pipeline re-runs from scratch** — fresh retrieval, fresh decomposition,
   the works — since the more likely problem here is that retrieval never
   found the right content for how the question was originally phrased, not
   that generation misused good content.

Both loops are bounded (`MAX_HALLUCINATION_RETRIES`, `MAX_QUESTION_REWRITES`,
default 1 each) — a persistently bad answer degrades to "best effort", never
an infinite loop.

**Verified two ways**: live, instrumented against real questions, both
checks genuinely execute on every turn (confirmed via a temporary wrapper
that logs each grade) — for an already-good answer, both pass on the first
try and the loop exits immediately, which is the expected, healthy case, not
dead code that never fires. Separately, since real answers rarely fail these
checks on demand (a sign the earlier layers are doing their job), the
*retry mechanics themselves* were verified against the real, live chain with
only the grading *decision* mocked (not the actions taken): forcing a
"hallucinating" grade on the first pass confirmed regeneration actually
calls `document_chain` with the same context, and forcing an "insufficient"
grade confirmed the question genuinely gets rewritten and fully re-retrieved
— both against real generation calls, not a fully mocked pipeline.

**A real off-by-one bug this surfaced**: the first version bounded the
hallucination-retry loop with `range(MAX_HALLUCINATION_RETRIES)` — with the
default of 1, that loop runs exactly once, meaning it could check-and-
regenerate but could never **re-check** whether the regeneration actually
fixed anything. A test built specifically to force a "fails once, passes on
retry" scenario caught this immediately (expected 2 grading calls, got 1).
Fixed to `range(MAX_HALLUCINATION_RETRIES + 1)` at first — later, merging
the two grading calls into one (`grade_answer`, see [Performance](#performance))
removed the awkward `+1` entirely: grading once before the loop and
re-grading after each regeneration is the shape that makes a plain
`range(N)` correct.

**Answer style: fuller and more natural, still strictly grounded.** The
original `ANSWER_PROMPT` ("precise research assistant... ONLY the context")
produced technically-correct but terse answers — often a single sentence
lifted close to verbatim from the source, even when the retrieved context
had more relevant detail sitting right there unused. Rewritten to explicitly
ask for full sentences and to include supporting detail already present in
the context (conditions, related specifics) rather than trimming to the
bare minimum — while keeping every other constraint (cite the real
`[source: ...]` tag, never invent one, say so if the context doesn't cover
it) unchanged. Verified on real data, not assumed: the same mission-
statement question that used to return one quoted sentence now returns two
paragraphs of actual explanation, every claim still cited to the same
source, and a minimal-content fixture correctly stayed terse rather than
padding with filler. The hallucination check above still grades the fuller
answer the same way, so the added length can't come from ungrounded claims
without getting flagged and regenerated.

## Evaluation

```bash
python scripts/evaluate.py
```

Runs the questions in `scripts/eval_dataset.json` through the full pipeline
and scores faithfulness, answer relevancy, context precision, and context
recall with RAGAS. The dataset is 16 real questions with ground-truth
answers pulled directly from the raw ingested text (not from the model's
own prior answers, to avoid the eval grading itself), spanning both PDFs,
the constitution, and 8 different web pages — expanded from an original 8,
deliberately drawing the new 8 from sources the first set never touched
(immersions, scholarships-and-loans, academics overview, graduate programs)
rather than just adding more of the same, so the expansion buys real
coverage breadth, not just row count.

**Real scores from a real run, all 16 questions scored**: faithfulness
0.93, answer relevancy 0.87, context precision 0.66, context recall 0.88 —
up from the original 8-question run's 0.84/0.86/0.77/1.00 on faithfulness
and relevancy, down on precision and recall. That's not noise to explain
away; reading the per-question breakdown (not just the mean) found a real,
concrete reason, on top of the two already-known ones below.

**Attacking the low precision score — and an honest non-result.** Context
precision (0.66) was the clear weak metric, so the first step was measuring
*why* rather than guessing. A quick data-driven scan (find lines repeated
verbatim across many pages — definitionally site chrome, not content) found
that **51% of all scraped web line-content was boilerplate**: the same ~100
nav-menu and footer-sitemap lines duplicated across all 16 ingested pages.
Every boilerplate-heavy chunk competing for a top-K retrieval slot is a slot
not spent on real content, so this looked like a direct, mechanical cause.

Fixed with `loaders.py`'s `strip_shared_boilerplate()`, detected from the
data rather than a hardcoded phrase list so it generalizes to any site, and
deliberately conservative (a line must appear on ≥50% of pages, and batches
under 4 pages are skipped entirely, since "1 of 2 pages" is 50% but means
nothing). Re-ingesting the same 16 URLs through it cut them from **398 to
208 chunks — a 48% reduction**, matching the measured 51% almost exactly.

Then the honest part: **re-running the full eval did NOT clearly pay off.**
Context precision moved 0.656 → 0.696 (+0.04, the right direction but
modest), while the other three metrics moved *down*: faithfulness 0.93 →
0.86, relevancy 0.87 → 0.82, recall 0.88 → 0.81. With only 16 questions and
LLM/retrieval non-determinism already demonstrated in this very project (the
TOEFL question scored context precision 0.5 in one run and 1.0 in the next
with *zero* code changes between them), a single row swinging moves any mean
by ~6 points — so these deltas can't be cleanly attributed to the
boilerplate change in either direction. Reporting the +0.04 as "fixed
precision" would be reading signal into noise.

The boilerplate stripping is kept anyway, on its own merits rather than on
this eval: 48% fewer chunks for identical real content is a measured,
verifiable win for index size, embedding cost, and retrieval-slot
competition, covered by five unit tests, whatever a 16-question eval's mean
happens to say that run. But "precision is fixed" is not a claim this run
supports, and the metric stays a known weak point.

**A flaw in the eval data itself, found by the same re-run**: the
"% international students" question's ground truth ("about 20%") came from
the academics-overview page's `20 / of our current students are
Internationals` — which has no `%` sign and is genuinely ambiguous. On the
re-run, retrieval surfaced a *different* RIT page stating **12%**
international, so the system answered 12% and scored 0 precision against my
own questionable reference. The system may well have been more right than
the eval. Worth flagging because it's the failure mode LLM-graded evals are
most prone to: a wrong-looking score that's actually a bad ground truth, not
a bad answer.

- **A genuine retrieval regression, caught by the eval, not assumed away**:
  the "two W's" scholarship-reduction question — part of the original 8,
  which scored context recall 1.00 as a whole then — now retrieves *zero*
  local documents and answers entirely from 5 generic US financial-aid
  websites (collegeraptor.com, a Wichita State KB article, UT Austin, Ohio
  State, bold.org) instead of the real, correct, still-present
  `WithdrawalForScholarships.pdf`. Confirmed directly by asking the exact
  question again outside the eval harness — same result, not a fluke. The
  document didn't go anywhere (verified: still 2 chunks in the store); the
  most likely cause is corpus growth diluting it out — the vector store
  grew from ~10 sources at the time of the original eval to 21 now
  (`minors`, `immersions`, `scholarships-and-loans`, `financial-office`,
  and others added since), several of which now also discuss
  scholarship-adjacent topics, and a 2-chunk PDF has little weight against
  a much larger, more crowded field competing for the same top-K retrieval
  slots. Corrective RAG's web fallback then did exactly what it's designed
  to do when local retrieval looks insufficient — it just triggered for
  the wrong reason here, on a question local content actually answers.
  Not fixed in this pass — recorded here as a genuine, current gap rather
  than silently patched or left undocumented: the honest fix is likely
  revisiting `RETRIEVER_TOP_K`/reranking sensitivity as the corpus keeps
  growing, not a one-off patch for this single question.
- **A citation-format gap in my own new code, found by the same run**:
  the answer to that same question cited its (wrong) web sources as bare
  `[https://...]`, not the `[source: https://...]` format
  `ANSWER_PROMPT` asks for. `pipeline.py`'s new `_cited_sources()` (see
  [Self-correcting generation](#self-correcting-generation)) originally
  required the exact `[source: ...]` prefix and would have silently
  under-reported real citations for any answer shaped like this one —
  fixed to accept a bare `[X]` too, both intersected against the real
  retrieved sources for safety. Found and fixed in the same pass this eval
  expansion surfaced it in, with a regression test.

Reading the per-question breakdown rather than just the mean also
reconfirmed the same two *phenomena* flagged in the original 8-question
run — worth being precise here rather than implying they're literally the
same rows scoring the same way twice, which they're not:

- The original run flagged one question at faithfulness 0.25 as an
  LLM-as-judge artifact on a short, clearly-correct answer (RAGAS's own
  faithfulness check is itself an LLM call, and can be inconsistent on
  short answers) rather than a real grounding failure. That exact row isn't
  reproducible run-to-run to check directly, but the same pattern showed up
  again here on a different question: "What is the required course for the
  International Relations Immersion?" (ground truth: a single short fact,
  "POLS-120, Introduction to International Relations") scored faithfulness
  **0.0** despite context precision 0.37 and context recall 1.00 — i.e. the
  right content WAS retrieved, which is inconsistent with a real
  hallucination and consistent with the same short-answer judge noise as
  before. Not re-verified line-by-line against the raw answer text this
  time, so held as "consistent with the known pattern," not re-proven from
  scratch.
- The original run flagged one question (TOEFL/IELTS scores) at context
  precision 0.5 because a step-back sub-question's independent web fallback
  leaked irrelevant context into the pool (see
  [Corrective RAG](#corrective-rag-c-rag)) — a real cost of decomposition +
  per-sub-question correction, not a one-off. That exact question actually
  scored a clean 1.0 on precision this run (LLM/retrieval non-determinism —
  the same question doesn't always decompose or fall back to the web the
  same way twice), but the same mechanism reproduced on two different
  questions instead: "the deadline for the Alternative Admission Process"
  and "what courses does the TDI offer" both scored context precision 0.5.
  Not individually root-caused per-row here — the point isn't that these
  exact two rows are special, it's that this leakage keeps showing up on
  *some* question every run, which is the real, load-bearing finding, not
  which specific question it lands on this time.

**A second real dependency bug found running this**: `ragas` (0.4.3, the
latest release) fails to import at all on the current `langchain-community`
— it unconditionally imports `langchain_community.chat_models.vertexai.
ChatVertexAI` just to list it in a static isinstance-check tuple (whether an
LLM supports multi-completion sampling, a feature this project doesn't use),
and that module no longer exists now that Vertex AI support moved to the
standalone `langchain-google-vertexai` package. Fixed with a small import
shim in `evaluate.py` — a harmless placeholder class registered in
`sys.modules` before `ragas` is imported, since the real class is never
actually instantiated, only checked against with `isinstance()`.

**A third real bug, plus a genuine methodology lesson**: the first real run
silently dropped one question's row from the final report entirely — no
error shown, just 7 rows instead of 8 — with `Exception raised in Job[6]:
TimeoutError()` buried in the log. RAGAS defaults to 16 concurrent workers;
that many simultaneous requests against Vertex AI's quota triggered a real
`429 ResourceExhausted`, and the retry/backoff for that under so much
concurrent contention compounded past even RAGAS's already-generous 180s
per-job timeout. Fixed by lowering concurrency (`RunConfig(max_workers=4,
timeout=240)`) — less simultaneous load means less to retry in the first
place, which is the actual fix; a higher timeout ceiling alone wouldn't have
addressed *why* jobs were slow. Confirming the fix actually worked took a
second pass, too: my own diagnostic — piping output through
`grep -v "...|return |..."` to strip noisy deprecation warnings — was blunt
enough to also strip the one row whose scraped web content happened to
contain the word "return" somewhere in it, producing a false "still only 7
rows" result on the first re-check. Re-verified against a completely
unfiltered capture before trusting the fix.

**A known, not-yet-fixed inefficiency, noticed running the expanded set**:
`run_eval()` builds a **fresh `ChatSession` per question** — the exact same
"measurement artifact" already caught and fixed in the latency-benchmarking
scripts (see [Performance](#performance)), reintroduced here independently
since this script was never touched during that fix. With 16 questions each
separately paying the ~13-90s cold-build cost, the eval run itself took
over 30 minutes. Deliberately not fixed in this pass: doing so safely means
reusing one session and clearing `.history` between questions (the same
pattern `bench.py` used), which would require re-running the full eval
again just to confirm nothing about the *results* changed — a real cost
worth paying at some point, but not worth spending on a pure speed
improvement to a script the results of this exact run already stand on.

## Retrieval tuning

Context precision was the weak metric, and [boilerplate stripping](#evaluation)
didn't clearly move it. The obvious next lever was `RETRIEVER_TOP_K` and
`RERANK_TOP_K` — but tuning them against the RAGAS eval would have been
useless, for a reason worth stating plainly: **that eval can't measure what
these knobs do.** It wraps retrieval in a router, scope gate, decomposition,
web fallback, generation, and an LLM-as-judge scorer, all non-deterministic.
A ±0.04 mean shift over 16 questions is indistinguishable from run-to-run
noise there — already demonstrated in this project, where the same question
scored context precision 0.5 in one run and 1.0 in the next with *zero* code
changes between them.

So the tuning got its own benchmark. **Retrieval itself is deterministic** —
embeddings, BM25, and cross-encoder reranking all return identical results
for identical input; the noise lives entirely in the LLM layers above. The
benchmark labels each of the 16 real eval questions with the source that
actually answers it, invokes *only* the local ensemble + reranker (no router,
scope gate, decomposition, web fallback, or generation), and measures what
fraction of returned chunks come from the right source. Zero LLM calls in the
loop, so a parameter sweep is genuinely apples-to-apples and repeatable.

Sweeping seven configurations:

| `RETRIEVER_TOP_K` | `RERANK_TOP_K` | precision | hit@k |
|---|---|---|---|
| 20 | 5 | 0.625 | 1.000 |
| **10** | **5** | **0.650** | **1.000** |
| 40 | 5 | 0.600 | 1.000 |
| 20 | 3 | 0.771 | 1.000 |
| 10 | 3 | 0.771 | 1.000 |
| 20 | 8 | 0.539 | 1.000 |
| 40 | 8 | 0.508 | 1.000 |

**The headline result is a trap, and catching it was the point.**
`RERANK_TOP_K=3` scores 0.771 — a 23% relative precision gain, by far the
best number in the table. Shipping it on that basis would have been wrong.
Precision is a *fraction*, so it rises trivially when you simply return
fewer chunks; the honest check is the absolute count of correct-source
chunks. Measured: going from 5 to 3 dropped **13 correct chunks (50 → 37)
across 10 of the 16 questions**. It wasn't removing noise, it was removing
signal — the denominator shrank. That would have made answers measurably
less complete while the metric looked better, and it would have quietly
undone the [fuller-answer work](#self-correcting-generation) done
deliberately earlier. Textbook Goodhart's law, caught only because the
absolute count was checked and not just the ratio. `RERANK_TOP_K` stays at
5.

**The real win was the unglamorous one.** `RETRIEVER_TOP_K` 20 → 10, with
`RERANK_TOP_K` untouched: precision 0.625 → 0.650, and the correct-chunk
count went **up**, 50 → 52 — three questions gained a correct chunk, one
lost one. Same five chunks still reach the LLM, so nothing is lost
downstream, and the initial-recall stage does less work (fewer candidates to
fetch and rerank). The mechanism makes sense in hindsight: a narrower
candidate pool gives the cross-encoder fewer near-miss distractors to
mistakenly rank above genuinely relevant content — more candidates is not
strictly better. 40 was worse on both counts, confirming the direction
rather than just the single step.

Honest scope: +0.025 precision is a small win, and 16 questions is a small
benchmark. It's a *real* win rather than a noise artifact — retrieval
determinism is what buys that claim — but it's not the fix that takes
precision from 0.66 to 0.9. The bigger remaining lever is corpus/chunking
quality, not these two knobs, which are now measured rather than guessed.

## Performance

Stacking every technique has a real cost: an in-scope question still fires
roughly a dozen LLM calls (router, contextualize, one scope check,
decompose, step-back, then per sub-question a conditional self-query +
relevance grade, then generation, then the answer grade). That is the
honest tradeoff of this project — it demonstrates each mechanism rather than
shipping the leanest possible pipeline.

**Measured, not guessed.** Profiling one real query (wrapping each step to
count calls and time) found ~24s of a ~51s query was serialized grader and
analysis calls — spread thin across many steps rather than concentrated in
one hotspot. Two things came out of that:

- A hypothesis that didn't survive contact with data: swapping graders to a
  smaller model. Benchmarked on Vertex, `gemini-2.5-flash-lite` returned in
  1.36s vs `gemini-2.5-flash` at 1.71s — far too small a per-call gap to
  explain the latency. The problem was the *number of sequential calls*, not
  the speed of each.
- A measurement artifact in my own test harness, worth flagging: the demo
  scripts built a **fresh `ChatSession` per question**, paying the ~13s
  one-time construction cost (loading every document for the BM25 index,
  the cross-encoder, the source catalog) on every single question. The CLI
  and API build it once and reuse it. Some of the "this is slow" was my
  benchmark, not the product.

**What actually changed:**

1. **Sub-questions retrieve concurrently** (`DecomposingRetriever` now uses
   `.batch()` with a bounded `RETRIEVAL_MAX_CONCURRENCY`). The whole
   per-question chain — scope gate, self-query, ensemble, rerank, relevance
   grading — runs in parallel across sub-questions instead of serializing.
   `.batch()` preserves input order, so merge/dedup stays deterministic.
2. **`decompose` and `step_back` run in parallel** — they never depended on
   each other.
3. **The two self-correction checks merged into one LLM call**
   (`grade_answer`). Both graded the same `(question, context, answer)`
   triple, so two round trips re-sent the expensive part (the context) for
   no extra signal. This also removed the awkward `+1` loop bound from the
   earlier off-by-one fix: grading once up front and re-grading after each
   regeneration makes a plain `range(N)` correct.

**Result** (session built once, as in real usage): **~67s → ~35s median per
question**, and the full test suite dropped from ~235s to ~138s on identical
workload. Concurrency is deliberately capped at 4 — Vertex AI returned
`429 ResourceExhausted` under heavier parallelism during eval runs.

**Two more targeted changes, on top of that:**

4. **The scope gate runs once, against the original question** — see
   [Scope gate](#scope-gate) for the mechanism. Verified directly, not
   assumed: profiling "What is the capital of France?" after the change
   shows exactly one `check_scope` call and zero `decompose`/`step_back`
   calls (11.7s total, vs. call counts that used to scale with sub-question
   count).
5. **Self-query construction is now conditional** — see
   [Self-query retrieval](#self-query-retrieval) for the mechanism.

Both are verified by unit tests (`tests/test_self_query.py`,
`tests/test_scope_gate.py`) and by the out-of-scope profiling run above, but
**did not show a clean wall-clock win on a small in-scope benchmark** — a
repeat of the 3-question benchmark from item 1-3 came back at 38.1s median,
not faster than the 35.4s before these two changes. That's expected, not a
failed optimization: all three benchmark questions are on-topic and share
real vocabulary with the corpus, which is exactly the case these two
changes are *not* supposed to skip anything for — and live Gemini API
latency varies by more per call than either change could plausibly save.
Their payoff is specific to out-of-scope questions and to questions whose
wording doesn't overlap the source catalog, not universal across every
question — worth having, but reported here for what it actually is rather
than rounded up to a bigger claim.

**A sixth change, and a real miss caught before it shipped:** query
decomposition/step-back is now conditional too, gated by a new
`needs_decomposition()` check in `query_analysis.py` — same pattern as the
scope gate, applied one level up. A simple, atomic, on-topic question skips
decompose + step-back entirely (two LLM calls) and the doubled retrieval
fan-out that comes with them, going straight to a single retrieval pass on
the original wording.

The first version of the gating prompt was close to useless: benchmarked
against 8 clearly-simple factual questions ("what is the tuition fee",
"when does the semester start"), it said "yes, decompose" for **7 of 8** —
barely better than always running it, while still adding its own LLM call
to every question. The prompt's "would broader context help" clause was too
permissive; almost anything can be argued to benefit from more context in
some general sense. Rewritten to require a specific, narrow trigger — either
a genuinely compound question, or one naming a specific entity (a course
code, a named program) whose answer more likely lives in a broader policy
section — re-benchmarked at **11/11 correct** on the same simple set plus a
compound set and two narrow-named-entity questions ("what's the drop
deadline for ACCT 110"), which correctly still trigger it. Would not have
caught this without actually running it against a batch of real questions
before treating the feature as done — a passing unit test (which only
checks the gate is *consulted*, not what a live model says) can't catch a
prompt that fails to discriminate.

Profiled directly on "What is RIT Kosovo's mission statement?": zero
`decompose`/`step_back` calls (previously always 2), 19.3s total query time.
The 3-question wall-clock benchmark used above doesn't show this cleanly,
for an honest reason specific to this change: two of those three questions
independently hit paths this change doesn't touch (a hallucination-triggered
regeneration on one, and a narrow-named-entity question that correctly
*keeps* full decomposition on the other) — the classifier's own call is a
small added cost paid by every question, compound or not, and only pays for
itself on the genuinely-simple ones. The per-question profiler evidence
above is the real signal here, not that median.

**A seventh change: the scope check and the decomposition check are now one
call, not two.** Items 4 and 6 above shipped as two separate upfront
classifiers — `check_scope()` deciding scope, `needs_decomposition()`
deciding decomposition — run sequentially, one after the other, on every
in-scope question. They're independent yes/no questions about the same
input, which is exactly the shape that already justified merging the two
self-correction checks into `grade_answer()` earlier in this list; the same
fix applies here. `check_scope_and_decomposition()`
([Scope gate](#scope-gate)) now answers both in one structured-output call,
and `ScopeGatedRetriever` dispatches directly on both answers — out of
scope, in-scope-simple, or in-scope-complex — instead of nesting a
decomposition-gated retriever inside a scope-gated one.

Verified directly, not assumed: profiling "What is RIT Kosovo's mission
statement?" (a simple, in-scope question) after the merge shows exactly one
`check_scope_and_decomposition` call and zero calls to the fallback-path
`needs_decomposition` — 4 tracked grader/analysis calls total (router,
combined gate, corrective relevance grade, answer grade), down from 5
before the merge. A compound question profiled the same way confirms the
complex path still dispatches correctly: the combined call fires once,
followed by real `decompose` and `step_back` calls and three
`grade_relevance` calls (one per resulting sub-/step-back question) — the
merge changes *how many calls decide what to do*, not what the pipeline
actually does once it's decided.

## Web UI internals

The [web UI](README.md#usage) ([`static/`](static/)) is plain HTML/CSS/JS —
no framework, no build step, no npm — mounted by FastAPI itself
([`api/main.py`](api/main.py#L23)) and calling `/query`/`/ingest` with
`fetch()`. Same origin as the API, so there's no base-URL config or CORS
setup to get right, and browser `fetch()` has no default timeout the way
Python's `requests` does — nothing to configure to survive a 30-90s query.
Chat history and `session_id` round-trip through the API exactly like the
CLI, so follow-ups ("what did you just say it was?") resolve correctly; the
session is also persisted to `localStorage`, so a page refresh doesn't lose
it. The "Cancel" button uses `AbortController` to actually stop the browser
from waiting — worth stating precisely rather than overselling it: the
backend has no cancellation hook, so the query keeps running server-side
regardless of what the UI does.

**Real per-stage progress, not a fake timer.** The obvious way to show
progress during a 30-90s query is a spinner with a generic "usually takes a
while" message. That was the first version — but a bare elapsed-time counter
cycling through canned text it can't actually verify is exactly the same
dishonesty as an overselling "Stop" button (see the Streamlit retrospective
below, which had exactly that problem). `/query/stream` sends the real thing
instead: newline-delimited JSON, one line per actual stage transition
(`{"stage": "retrieving"}`, `{"stage": "generating"}`, ...) as
`ChatSession.ask()` really moves through them, plus a final line with the
answer. `pipeline.py`'s `on_stage` callback fires at each real transition —
routing, retrieving, generating, verifying, and (only when they actually
happen) regenerating or rewriting the question.

Splitting "retrieving" from "generating" needed real verification, not
assumption: `create_retrieval_chain` runs retrieval and generation as one
opaque call, so "generating" has to come from a LangChain
`on_retriever_end` callback firing at the true retrieval-finished moment —
but a custom `BaseRetriever`'s nested calls to OTHER custom retrievers
(exactly what `DecomposingRetriever`'s fan-out across sub-questions does)
turned out to propagate the ambient callback manager when invoked as part
of a composed chain, producing multiple `on_retriever_start`/`end` pairs,
not the single pair a bare `retriever.invoke()` call produces in isolation.
Confirmed both behaviors directly with throwaway scripts before writing
`_RetrievalDoneCallback` — a depth counter that only fires once retrieval
work, including all nested fan-out, has actually finished — rather than
trusting the first assumption. Also caught the same way: building a
brand-new `ChatSession` (~13-58s cold) used to happen before the stream
even started, so the client would sit showing "Routing..." while the
server hadn't reached routing yet — moved that construction inside the
background thread too, behind its own real `"initializing"` stage, once
watching it live in a browser showed the mislabeled wait directly.

**An earlier version of this UI was built in Streamlit instead.** Working
Python-only UI, but every interaction reran the whole script (the sidebar's
health check refired on every click), its own "Stop" control couldn't
actually interrupt a blocking request either — same limitation as
`AbortController` above, just presented misleadingly as if it could — and
its chrome (default theme, Deploy button, hamburger menu) wasn't something
this UI fully controlled. Replaced with the hand-written version above once
it became clear this should look like a distinct product, not a generic
demo shell.

**A real concurrency bug, found by actually running this, not by
inspection:** the first few requests against a freshly started API process
intermittently crashed `/health` and `/query` with a `KeyError` /
`AttributeError` buried inside chromadb's rust bindings. `get_chroma_client()`
was `@lru_cache`-wrapped as a "one shared client per process" singleton — but
`lru_cache` only makes the cache dict's own read/write atomic, not the
"miss, call the function, store the result" sequence around it. FastAPI runs
sync endpoints in a thread pool, so two requests arriving close together
could both see a cache miss and both call `chromadb.PersistentClient(...)`
for the same path at once — chromadb's own shared-system registry isn't safe
against that, and the second caller got a corrupted, unrecoverable client
for the rest of the process. Reproduced live by firing 5 truly simultaneous
requests at a fresh server; fixed with a hand-rolled double-checked-locking
singleton in [`vectorstore.py`](src/rag/vectorstore.py) instead, and covered
by a deterministic regression test in
[`test_vectorstore.py`](tests/test_vectorstore.py) that stubs the
constructor with an artificial delay to force the same overlap on demand
rather than relying on timing luck.

## Notes on design choices

- **Chroma** over a hosted vector DB: zero infra, persists to disk, plenty for
  a single-tenant project. Swapping in Pinecone/Weaviate would only touch
  `src/rag/vectorstore.py`.
- **BM25 + vector hybrid**: pure embedding similarity misses exact-match terms
  (error codes, config keys, proper nouns) that keyword search catches; fusing
  both is a standard production pattern, not just an academic nicety.
- **Local cross-encoder reranker**: no extra paid API, keeps latency and cost
  down, and is a well-known technique for improving precision@k after a wide
  initial recall pass.
- **Web search as a graded fallback, not an always-on ensemble member**: an
  earlier version blended it into every retrieval unconditionally — simpler,
  but not actually Corrective RAG, and it spends a Tavily call on every
  question even when local docs already answer it fine. Grading first and
  falling back only when needed is both more correct and cheaper.
- **Text-to-SQL kept as a genuinely separate path, not bolted onto RAG**:
  flattening structured data to text (the vector-store path) and keeping it
  relational (the SQL path) solve different problems — lookups vs.
  computation — and pretending one pipeline does both well would be the
  actual design mistake.
- **Router skips the LLM call when there's nothing to route to**: if no SQL
  database exists, classifying the question is pure overhead — the answer is
  always "vectorstore." Cheap short-circuits like this matter more as more
  routes get added.
- **Rerank per sub-question, not once against the combined question**: this
  was a real bug, not a hypothetical one — see
  [Query analysis](#query-analysis) for what broke and why the fix
  is to rerank before merging, not after.
- **Grade relevance per sub-question too, for the same reason**: correction
  is wrapped inside decomposition, not around the merged result, so one weak
  sub-question triggers its own web fallback without forcing every other
  (already well-answered) sub-question to pay for one too.
- **Self-query scoped to `source`/`sheet` only, not arbitrary CSV/JSON
  columns**: those columns' names and meaning vary per ingested file and
  can't be described generically up front the way `AttributeInfo` needs — and
  filtering on values that vary per computation is what text-to-SQL is for
  anyway. The two `source`/`sheet` values that *are* declared are grounded in
  the real distinct values seen in the store, not left for the LLM to guess
  at plausible-sounding filenames.
- **A real dependency/compatibility bug worth knowing about**: wiring up
  `SelfQueryRetriever` surfaced two: (1) its automatic vectorstore-translator
  detection eagerly imports ~20 other vector-store integrations to guess
  which one applies, and on the installed `langchain-community` version one
  of those imports (`DatabricksVectorSearch`) no longer exists — so the whole
  detection fails even though the vectorstore in use (Chroma) is fine. Fixed
  by passing `ChromaTranslator()` explicitly instead of relying on
  auto-detection. (2) The query-constructor's output parser needs the `lark`
  package, which isn't a `langchain` dependency — it has to be installed
  separately.
- **Parent-document retrieval kept as its own Chroma collection, not folded
  into the main one**: child chunks carry a `doc_id` pointing into a separate
  docstore, a different metadata shape than `source`/`sheet`/etc. — mixing
  them would make every other retriever's metadata assumptions unreliable.
- **One shared `chromadb.PersistentClient` across every collection, not one
  per `Chroma(...)` call**: this was a real, reproduced bug — adding a second
  collection (parent-document) sharing the main one's persist directory
  caused an intermittent `Nothing found on disk` error from separate client
  objects racing on the same on-disk store. `vectorstore.get_chroma_client()`
  is cached once per process and reused everywhere a Chroma collection is
  opened. (A second, subtler race in this same client — concurrent *first*
  construction under FastAPI's thread pool — was found and fixed later; see
  [Web UI internals](#web-ui-internals).)
- **The most significant bug found in this whole project**: the generation
  prompt instructed the model to "cite sources using `[source: <name>]`", but
  `create_stuff_documents_chain` was never given a `document_prompt` — its
  default document formatting shows the LLM ONLY `page_content`, never
  metadata. So the model was never actually shown a real source name to cite
  — it was inventing plausible-looking ones from whatever text happened to be
  in the chunk. This worked *by coincidence* most of the time (many chunks
  happen to open with a title that matches the real filename closely enough
  to look right), which is exactly why it went unnoticed through many rounds
  of manual verification — until a parent-document chunk from the middle of
  the handbook (no title in view) got cited as `[source: 20]`, an internal
  clause number from the document's own text, and a web-search chunk with no
  clean title got cited as `[source: <the entire first sentence of the
  passage>]`. Fixed by giving `create_stuff_documents_chain` an explicit
  `document_prompt` (`chain.py`) that tags every passage with its real
  `source` metadata before the model ever sees it, and rewording the prompt
  to say "copy the exact tag shown," not "cite a source." Every citation
  produced since matches real metadata exactly — worth flagging because it's
  a good example of how a plausible-looking LLM output can pass repeated
  manual spot-checks while being subtly ungrounded underneath.
- **One source catalog, shared by two components, not two separate
  summarization mechanisms**: self-query and the scope gate both need "what
  does this source roughly cover" — generating it once
  (`source_catalog.py`) and having both read the same cached file is simpler
  and cheaper than each inventing its own, and keeps them from ever
  disagreeing about what a source is about.
- **Scope gate is a latency/cost optimization, not a correctness mechanism —
  Corrective RAG stays the real safety net**: the gate only ever looks at
  short static descriptions, never actual content, so it's structurally
  unable to catch "this looked relevant but the retrieved content wasn't
  actually useful" — only Corrective RAG's post-retrieval grading can. The
  gate is deliberately biased toward false positives (default to "in scope"
  when unsure) because a wrongly-skipped local retrieval is a real
  correctness regression, while a wrongly-run one just costs one extra pass
  that Corrective RAG already handles correctly.
- **`ChatSession.ask()` catches failures from the SQL path, not just from
  individual retrievers**: every retriever-level component in this project
  (`SafeRetriever`, the corrective/scope-gate fallbacks) already degrades
  instead of crashing — but a real live run found that the *session* level
  didn't have the same guarantee. A malformed SQL query correctly got
  rejected by the safety guard, and that rejection correctly raised an
  exception — but nothing caught it, so it crashed the whole session instead
  of just that one answer. Fixed by falling back to the RAG path for the
  same question on any SQL-path failure, matching the "skip, don't crash"
  posture everything else in this project already had.
- **Regenerate on hallucination, but re-retrieve on "doesn't answer"** —
  deliberately different corrective actions for the two self-correction
  checks, not the same retry logic reused twice. Retrieval is deterministic:
  re-running it with the same question after a hallucination would return
  the identical context and likely the identical answer, so that fix has to
  happen at generation time (regenerate from the same context, with
  feedback). An answer that doesn't resolve the question is a different
  failure shape — the more likely cause is retrieval never finding the right
  content for how the question was phrased, so the fix there is rewriting
  the question and re-running retrieval from scratch, not touching
  generation at all.
- **`document_chain` exposed separately from the full retrieval chain**:
  needed so hallucination-triggered regeneration can reuse the already-
  retrieved context directly, instead of the only alternative being a full
  (and, per the point above, pointless) re-retrieval. A small, deliberate
  crack in encapsulation for a real, specific need — not exposed for its
  own sake.
- **The hallucination-retry loop bound was a real off-by-one bug, not a
  hypothetical one**: `range(MAX_HALLUCINATION_RETRIES)` let the loop
  regenerate once but never re-check the regenerated answer — caught by a
  test built specifically to exercise a "fails once, passes on retry"
  scenario, which is exactly the case that bug silently broke. See
  [Self-correcting generation](#self-correcting-generation) for the fix. The
  later merge of the two graders removed the awkward `+1` entirely: grading
  once before the loop and re-grading after each regeneration is the shape
  that makes a plain `range(N)` correct.
- **Profile before optimizing — one hypothesis died on contact with data**:
  the obvious latency fix looked like "run the graders on a cheaper model,"
  but benchmarking showed only a 1.36s vs 1.71s per-call gap. The real cost
  was serialization, fixed with concurrency instead. See
  [Performance](#performance).
- **Assert on the mechanism, not on output text**: the regeneration test
  originally asserted the regenerated answer *differed* from the original.
  It didn't — at near-zero temperature over identical context and a
  one-sentence fixture, a correct regeneration reproduces the same wording
  verbatim. The assertion was wrong, not the code. Rewritten to spy on
  `document_chain` and assert it was invoked with the already-retrieved
  context, which is what the test actually claims to verify.
- **A real bug found by auditing for unused code, not by anyone hitting
  it**: `requirements.txt` listed `python-docx`, but nothing in the app ever
  imports it — `.docx` loading goes through langchain's `Docx2txtLoader`,
  which needs the differently, confusingly-named `docx2txt` package
  instead. That package was missing entirely. Dropping a real `.docx` into
  `data/raw` would have crashed ingestion outright
  (`ModuleNotFoundError: No module named 'docx2txt'`) — undetected because
  no test exercised `.docx` loading at all, only PDF/MD/CSV/JSON/Excel.
  Fixed by swapping the dependency and adding
  `test_load_directory_handles_docx` in
  [`test_ingestion.py`](tests/test_ingestion.py), following the same
  build-the-fixture-on-the-fly pattern as the existing Excel test.
  `python-docx` is still a (test-only) dependency — it's what authors the
  `.docx` fixture — but the app's own runtime path never imports it, which
  is now stated explicitly in `requirements.txt` rather than left
  ambiguous. `unstructured` and `markdown` were dropped from
  `requirements.txt` too: leftover from an earlier iteration of the
  loaders, no longer imported by anything.
- **A smaller one from the same audit**: `self_correction.py` imported
  `src.config` and never used it — a leftover from before `grade_answer`'s
  merge refactor moved the retry-count logic into `pipeline.py`. Removed.
- **A gate that didn't gate**: the first version of `needs_decomposition()`'s
  prompt said "yes, decompose" for 7 of 8 clearly-simple test questions —
  functionally always-on, while still charging every question an extra LLM
  call. Unit tests only proved the gate was *consulted*, never that a live
  model actually discriminated with it; that only showed up by running it
  against a real batch of questions. See
  [Performance](#performance) for the fix and the re-verified numbers.
- **The biggest one: ingestion had no dedup, and it silently corrupted the
  real store**. Neither `add_documents()` (main collection) nor
  `ingest_parent_documents()` (parent-document index) ever passed explicit
  Chroma ids, so every chunk got a fresh random one on every call — meaning
  re-ingesting the SAME file added it again instead of updating it in
  place. This had been running invisibly the whole project: a routine
  request to add two new URLs (which also re-scans `data/raw/`
  unconditionally, an existing behavior of `cmd_ingest`) surfaced that
  **1340 of 1515 chunks in the main collection, and all 933 in the
  parent-document index, were exact duplicates** — accumulated over
  however many times `ingest` had quietly been re-run across the project's
  life. Every duplicate chunk is a wasted slot in reranking and relevance
  grading, diluting results without adding any real content. Fixed at the
  root — `chunking.py` and `parent_document.py` now assign a deterministic
  id from `(source, exact content)`, via a new `_stable_chunk_id`/
  `_stable_source_doc_id` helper — so re-ingesting the same content upserts
  instead of duplicating, covered by two new regression tests. The
  *existing* duplication was real data, not a hypothetical, so it needed a
  real fix too: a one-off surgical cleanup (group by exact `(source,
  content)`, keep one copy, delete the rest — no re-embedding, no API
  calls) brought the main collection to 765 chunks and the parent index to
  311, both verified back to a working, correctly-cited answer afterward.
  One acknowledged gap: `ParentDocumentRetriever`'s CHILD chunks still get
  LangChain-internal random ids (not exposed for override without
  reimplementing its private splitting logic) — lower-impact than the bug
  above, since duplicate child embeddings still resolve to the same parent
  text, which `DecomposingRetriever`'s final merge already dedupes by
  `(source, content)` before it reaches the model.
- **A follow-on bug the boilerplate fix exposed immediately**: content-hash
  chunk ids handle *re-ingestion* collisions correctly (that's the point —
  same content upserts instead of duplicating), but never had to handle two
  chunks colliding **inside one batch**. Stripping shared nav/footer lines
  reduced several thin pages to byte-identical residual text, so they hashed
  to the same id in a single ingest call — and Chroma rejects a batch
  containing a duplicate id outright (`DuplicateIDError`) rather than
  treating it as an upsert, failing the entire run. Caught the first time
  the re-ingest was actually executed, not by inspection. Fixed by deduping
  by id in `vectorstore.add_documents()` before adding, which is the correct
  behavior regardless: two chunks with the same content hash *are* the same
  chunk by this project's own definition of chunk identity. Two regression
  tests, including one confirming chunks with no id (`None`) are never
  collapsed together.
