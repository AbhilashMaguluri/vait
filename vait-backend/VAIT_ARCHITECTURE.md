# VAIT Institutional Knowledge Architecture & Multi-Tier Retrieval Pipeline
*The Authoritative Architecture & Engineering Specification for VAIT (Vasireddy Venkatadri AI Assistant)*

---

## 1. Executive Summary & Dual-Period Architecture

VAIT is designed with institutional grounding for **Vasireddy Venkatadri Institute of Technology (VVIT)** and its evolution into **VVIT University (VVITU)**.

### Dual Institutional Periods
1. **Current (2024–Present):** **VVIT University (VVITU)**
   - Official domain: `vvitu.ac.in` (and `www.vvitu.ac.in`)
   - Technology stack: Modern client-side React SPA, dynamic client-side routing, asynchronous API hydration.
2. **Historical (2007–2024):** **Vasireddy Venkatadri Institute of Technology (VVIT)**
   - Official legacy domain: `vvitguntur.com` (and `www.vvitguntur.com`)
   - Technology stack: Server-rendered CMS (Joomla), static archives, legacy syllabi, historical leadership.

---

## 2. Four-Tier Retrieval Pipeline

VAIT organizes retrieval into four distinct, prioritized tiers to maximize speed, accuracy, and fallback resilience while strictly preventing hallucinations.

```mermaid
graph TD
    UserQuery["User Query"] --> Tier1["Tier 1: FAISS RAG (Local Vectors)"]
    Tier1 -- "Adjusted score >= 0.70 & Entity present" --> ContextPrep["Context Assembly & Generation"]
    Tier1 -- "Refusal / Score < 0.70 / Entity Missing" --> WebDiscovery["Official Web Discovery"]
    
    WebDiscovery --> URLCheck{"Direct URL or Discovered Route"}
    URLCheck --> Tier2["Tier 2: Fast HTTP Fetch (~200ms)"]
    
    Tier2 -- "Valid Server HTML & >= 120 chars" --> QualityCheck["Evidence Quality Validator"]
    Tier2 -- "Client-side SPA / Dynamic JS Shell" --> Tier3["Tier 3: Headless Chromium DOM (~1.5s - 4s)"]
    
    Tier3 -- "Valid Rendered DOM / Entities" --> QualityCheck
    Tier3 -- "Insufficient text (<120 chars) / Canvas / Obfuscated" --> Tier4["Tier 4: Query-Aware Semantic Screenshot + Vision (~3s - 5s)"]
    
    Tier4 -- "Vision OCR Extraction (Primary/Secondary)" --> QualityCheck
    QualityCheck -- "Valid Evidence" --> ContextPrep
    QualityCheck -- "Invalid / Failed" --> DirectURL["Tier 5 Fallback: Direct Verified Official URL"]
```

### Tier Specifications

| Tier | Subsystem | Latency | Technology | Scope & Purpose |
|---|---|---|---|---|
| **Tier 1** | Local Vector RAG | 0ms – 150ms | FAISS index, local embeddings | Core institutional handbook, static rules, admissions criteria |
| **Tier 2** | Fast HTTP Fetch | 150ms – 400ms | `requests` / `httpx`, BeautifulSoup | Static server-rendered pages, legacy archives, Joomla endpoints |
| **Tier 3** | Headless Browser DOM | 1.5s – 4.0s | Playwright Chromium, SPA hydration | React SPA rendered DOM, card grids, HTML tables |
| **Tier 4** | Query-Aware Screenshot + Vision | 3.0s – 5.0s | Playwright semantic tiling + Multimodal LLM OCR | Canvas rendering, visually-nested cards, image banners |
| **Fallback** | Direct Canonical URL | 0ms | Curated official route mapping | Direct clickable link provided when automatic extraction fails |

---

## 3. Query-Aware Semantic Screenshot & Vision Subsystem (Tier 4)

### Bounded Element Targeting & Semantic Tiling
- **The Problem:** The root container `#root` on modern SPAs can span upwards of 10,000 pixels. Capturing full-page screenshots squashes typography, exceeding multimodal vision token limits and rendering text illegible.
- **The Solution:**
  1. **Generic Query-Aware Scoring:** Analyzes DOM candidate containers (`main`, `article`, `[class*='grid']`, `table`, `.container`) and scores them using query terms, semantic tags (`h1`-`h4`), and structured class keywords.
  2. **Bounded Element Screenshot:** If target element height $\le 1600	ext{px}$, captures a single high-resolution PNG.
  3. **Semantic Tiling:** If target element height $> 1600	ext{px}$, partitions child elements (`cards`, `sections`, `rows`) into up to 4 bounded tiles.
  4. **Multi-Tile Extraction & Deduplication:** Runs multimodal vision over individual tiles and merges/deduplicates extracted facts.

### Dual-Model Vision Fallback
1. **Primary Model:** `google/gemini-2.5-flash` via OpenRouter (low latency, high OCR visual accuracy).
2. **Secondary Model:** `meta-llama/llama-3.2-11b-vision-instruct` via OpenRouter (robust secondary fallback).
3. **Extraction Prompt Guard:** Instructs the vision model to extract only explicitly visible facts and output strict negative notice if information is absent, preventing visual hallucination.

### Memory & Cache Safety
- Raw Base64 screenshot strings are discarded immediately following text extraction.
- The returned `BrowserRenderResult` and `OfficialWebResult` store `screenshot_b64 = None` to ensure zero memory bloat in memory caches.

---

## 4. Security & Guardrails (SSRF & Concurrency)

### SSRF Protection Model
- **Strict Domain Allowlist:** Only `vvitu.ac.in`, `www.vvitu.ac.in`, `vvitguntur.com`, and `www.vvitguntur.com` are permitted.
- **Private IP & Loopback Filter:** Immediate rejection of `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`, `0.0.0.0`, and IPv6 equivalents.
- **Playwright Route Interception:** Client-side JavaScript redirects and navigation events are intercepted via `page.route("**/*")`. Any navigation targeting off-domain hosts is aborted (`blockedbyclient`).

### Concurrency Architecture
- **Initialization Lock (`_init_lock`):** Serializes Playwright browser launching and teardown.
- **Request Semaphore (`_semaphore`):** Governs concurrent page operations via `asyncio.Semaphore(max_concurrent_browser_pages)` (default: 3).

---

## 5. Caching & Evidence Validation

### Content-Aware Cache TTL
- **Static Institutional Records:** 7,200 seconds (2 hours) for official catalogs, leadership rosters, legacy archives.
- **Dynamic Content:** 600 seconds (10 minutes) for live headless browser renders, examination circulars, notifications.

### Evidence Quality Validation
- Implemented via `EvidenceQualityValidator`:
  - Classifies query intent: `faculty`, `exam`, `fee`, `course`, `notice`, `admission`, `general`.
  - Verifies factual density: requires structured entities or domain keywords and minimum text lengths.
  - Rejects extraction failure placeholders and navigation headers.

---

## 6. Cloud Deployment Topology

- **Containerization:** Production Docker image based on `python:3.11-slim-bullseye`.
- **System Dependencies:** Includes all required X11, Cairo, Pango, and NSS shared libraries for headless Chromium execution.
- **Non-Root Execution:** Runs under dedicated `appuser` (UID 1000).
- **Configuration Management:** Environment variables configured in `render.yaml` and loaded via Pydantic `Settings`.

---

## 7. Conversational Context & Multi-Turn Reference Resolution

VAIT implements a generalized multi-turn reference resolution engine (`ConversationContextResolver`) to maintain continuity across follow-up queries without hardcoding specific subjects or conversational shortcuts.

```mermaid
graph TD
    UserTurn["Follow-up Query (e.g. 'fetch details about them')"] --> Resolver["ConversationContextResolver"]
    Resolver --> Classify["Analyze Query Tokens & Pronouns"]
    Classify -- "Plural Reference (them, they, these, all of them)" --> MultiEntity["Resolve to Current Tracked Entities (61 AI Faculty)"]
    Classify -- "Ordinal Reference (first one, second, last)" --> SingleEntity["Resolve to Specific Tracked Entity by Position"]
    Classify -- "Department Transition (what about ECE?)" --> Transition["Transition Domain & Reset Entities"]
    
    MultiEntity --> EvidenceCheck{"Is Already-Retrieved Evidence Sufficient?"}
    EvidenceCheck -- "Yes (Shallow Query)" --> ReuseEvidence["Reuse Institutional Evidence Immediately (0 network calls)"]
    EvidenceCheck -- "No (Deep Query e.g. publications/bio)" --> DeepFetch["Fetch Deeper Profiles (up to max_deeper_profiles_fetch)"]
    
    SingleEntity --> PromptAugment["Augment Query with Grounded Entity Context"]
    ReuseEvidence --> PromptAugment
    DeepFetch --> PromptAugment
    Transition --> RAGRetrieval["Perform Fresh Official Retrieval"]
    
    PromptAugment --> LLMGen["Ground in Institutional Context (Prevent Generic Refusal)"]
```

### Reference Resolution Model
1. **Generic Linguistic Tokens:** Resolves plural references (*them, they, these, those, their*), singular references (*he, she, it*), and ordinal references (*first, second, third, last*) using clean tokenization. Zero hardcoded query strings.
2. **Multi-Format Entity Extraction:** Extracts structured entities (`ExtractedEntity`) from:
   - Structured markdown tables generated from dynamic web grids.
   - Structured JSON card structures (`res.entities`).
   - Bulleted entity lists and profile URLs.
3. **Shallow vs. Deeper Profile Crawling:**
   - **Shallow follow-up queries** (e.g., "fetch details about them", "give me more info"): The resolver identifies that the already-retrieved institutional evidence contains detailed tables/cards. It reuses the evidence context directly, avoiding redundant network trips.
   - **Deep follow-up queries** (e.g., "what are Dr. Suresh's publications?"): The resolver inspects entity `profile_url` attributes and selectively crawls deeper profile pages up to `max_deeper_profiles_fetch` (default: 3).
4. **State Persistence & Hydration:**
   - Conversation context state (`ConversationContext`) is serialized and persisted to MongoDB via `record_chat_exchange`.
   - On subsequent turns, `_get_or_rehydrate_context` dynamically reconstructs context from either the stored `context_state` or the multi-turn history window (`max_context_history_turns = 10`).
5. **Streaming & Non-Streaming Parity:**
   - Both `/chat` and `/chat/stream` share identical reference resolution, evidence reuse, entity extraction, and state persistence workflows.

---

## 8. Indian Standard Time (IST / Asia/Kolkata) Grounding

VAIT enforces Indian Standard Time (`Asia/Kolkata`, UTC+05:30) project-wide to ensure temporal accuracy for academic schedules, circulars, admission deadlines, and date calculations.

### Backend Timezone Enforcement
- **Central Configuration:** `app_timezone: str = "Asia/Kolkata"` defined in `Settings` with automatic fallback via the `tzdata` package.
- **Grounding Helper:** `app.utils.timezone` provides `now_ist()`, `to_ist()`, `format_ist()`, `resolve_relative_date()`, and `get_ist_grounding_context()`.
- **System Prompt Injection:** Injects current IST date, day of week, time, and academic year into LLM prompts (`_build_generation_prompt` and `_build_llm_only_prompt`):
  ```
  CURRENT SYSTEM TIMEZONE & DATE (Asia/Kolkata / IST, UTC+05:30):
  - Current Date & Time: Saturday, 12 September 2026, 10:30 AM IST
  - Current Day: Saturday
  - Indian Standard Time (IST) offset: +05:30
  - All relative date inquiries ('today', 'yesterday', 'tomorrow', 'this week', 'current academic year') MUST be interpreted strictly according to this IST reference date.
  ```

### Frontend Timezone Formatting
- All user-facing timestamps in `MessageBubble.jsx` and `HistoryList.jsx` explicitly use `timeZone: 'Asia/Kolkata'` in `toLocaleTimeString` and `toLocaleDateString`, guaranteeing visual consistency regardless of the user's browser locale.

