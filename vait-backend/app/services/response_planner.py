"""
VAIT Response Planner — Intelligent adaptive response classification and formatting.

Determines the most effective presentation structure based on the question type:
  - simple: compact direct answer without unnecessary cards or padding
  - factual: prominent highlighted hero fact with era tag and concise context
  - list: clean structured list with icon bullets or compact cards
  - comparison: side-by-side comparison table with feature attributes
  - educational: 5-stage teaching structure (definition, intuition, mechanics, example, exam summary)
  - academic_exam: 10-mark exam structure (definition, working, algorithm/steps, advantages/limitations, summary)
  - process: numbered step-by-step procedure or timeline
  - checklist: requirement checklist with checkboxes (- [ ])
  - schedule: date/calendar timetable
  - fee: structured fee table with current VVITU vs historical VVIT segregation
  - calculation: step-by-step formula evaluation (Formula -> Substitution -> Calculation -> Final Answer)
  - code: syntax-highlighted programming script with language indicator
  - historical: archival narrative of legacy VVIT institute
  - current_historical: institutional transition flow (VVIT -> Transition -> VVITU)
  - warning: policy notices, deadlines, or missing info alerts
  - no_answer: concise institutional refusal
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ResponsePlan:
    """Plan specifying the intended response format and dynamic LLM directives."""
    response_type: str
    title_hint: str
    formatting_instructions: str
    layout_density: str = "medium"  # "compact" | "medium" | "expanded"
    highlight_candidate: Optional[str] = None


class ResponsePlanner:
    """Analyzes user queries and formulates optimal presentation blueprints."""

    # ── Compiled Regular Expression Matchers ─────────────────────────────────

    _RE_COMPARISON = re.compile(
        r"(?i)\b(compare|comparison|versus|\bvs\b|diff(?:erence)?\s+between|which\s+is\s+better)\b"
    )

    _RE_CODE = re.compile(
        r"(?i)\b(write\s+(?:a\s+)?(?:python|java|c\+\+|c|javascript|sql|code|program|script|function)|"
        r"code\s+for|algorithm\s+for|implement\s+(?:in|a|the)|pseudocode)\b"
    )

    _RE_CALCULATION = re.compile(
        r"(?i)\b(calculate|percentage\s+of|compute|solve|formula\s+to\s+calculate|how\s+to\s+calculate|"
        r"score\s+calculation|cgpa\s+calculation|math\s+behind)\b"
    )

    _RE_CHECKLIST = re.compile(
        r"(?i)\b(documents?\s+(?:required|needed|to\s+submit)|prerequisites|eligibility\s+checklist|"
        r"checklist|what\s+documents|certificate\s+requirements|things\s+to\s+carry)\b"
    )

    _RE_PROCESS = re.compile(
        r"(?i)\b(how\s+does\s+(?:admission|counseling|registration|examination|placement)\s+work|"
        r"how\s+to\s+apply|admission\s+(?:process|procedure|steps)|step\s*by\s*step|"
        r"procedure\s+for|process\s+of|how\s+can\s+i\s+(?:get|apply|register|join))\b"
    )

    _RE_SCHEDULE = re.compile(
        r"(?i)\b(academic\s+calendar|exam\s+(?:dates?|schedule|timetable)|when\s+(?:is|are|does|do)|"
        r"semester\s+start|commence|holidays?|vacation|schedule\s+for|important\s+dates?)\b"
    )

    _RE_FEE = re.compile(
        r"(?i)\b(fee\s+(?:structure|details?|payment|concession)|tuition\s+fee|hostel\s+fee|"
        r"bus\s+fee|how\s+much\s+is\s+the\s+fee|examination\s+fee|cost\s+of)\b"
    )

    _RE_ACADEMIC_EXAM = re.compile(
        r"(?i)\b(10[\s-]marks?|16[\s-]marks?|exam\s+answer|long\s+answer|essay\s+answer|"
        r"detailed\s+notes\s+on|exam\s+preparation\s+for)\b"
    )

    _RE_EDUCATIONAL = re.compile(
        r"(?i)\b(explain|teach\s+me|what\s+is|how\s+does\s+.+\s+work|understand|learn|"
        r"concept\s+of|working\s+of|architecture\s+of|overview\s+of|theory\s+of)\b"
    )

    _RE_TRANSITION = re.compile(
        r"(?i)\b(what\s+happened\s+to\s+vvit|transition\s+from\s+vvit|vvit\s+to\s+vvitu|"
        r"is\s+vvit\s+(?:now|became)\s+vvitu|vvit\s+vs\s+vvitu|history\s+of\s+university\s+status)\b"
    )

    _RE_HISTORICAL = re.compile(
        r"(?i)\b(in\s+20\d\d|history\s+of\s+vvit|when\s+was\s+vvit\s+founded|legacy\s+vvit|"
        r"older\s+vvit|past\s+vvit|former\s+vvit)\b"
    )

    _RE_LIST = re.compile(
        r"(?i)\b(what\s+are\s+the\s+(?:[a-z0-9\.\s-]+)?(?:branches|courses|departments|programs|degrees|clubs|facilities)|"
        r"list\s+(?:of\s+)?(?:all\s+)?(?:[a-z0-9\.\s-]+)?(?:branches|courses|departments|programs|degrees|clubs|facilities)|"
        r"name\s+the\s+(?:[a-z0-9\.\s-]+)?(?:branches|courses|departments|programs)|"
        r"available\s+(?:branches|courses|programs|seats))\b"
    )

    _RE_FACTUAL = re.compile(
        r"(?i)\b(who\s+is\s+(?:the\s+)?(?:current\s+)?(?:principal|vice[\s-]chancellor|dean|chairman|director|hod)|"
        r"what\s+is\s+(?:the\s+)?(?:current\s+)?(?:official\s+)?(?:website|portal|address|contact|phone|email|location)|"
        r"where\s+is\s+(?:the\s+)?(?:vvit|vvitu|college|campus)(?:\s+located)?)\b"
    )

    _RE_SIMPLE = re.compile(
        r"(?i)^(hi|hello|hey|good\s+(?:morning|afternoon|evening)|what\s+is\s+vait|"
        r"who\s+are\s+you|what\s+does\s+vait\s+stand\s+for|what\s+can\s+you\s+do|help)\b"
    )

    @classmethod
    def plan_response(
        cls,
        query: str,
        intent: str = "general",
        period: str = "both",
    ) -> ResponsePlan:
        """Analyze the query and determine the optimal ResponsePlan."""
        q = query.strip()

        # 1. Simple Greetings & Identity Inquiries
        if cls._RE_SIMPLE.search(q) or len(q.split()) <= 4 and cls._RE_SIMPLE.match(q):
            return ResponsePlan(
                response_type="simple",
                title_hint="Direct Overview",
                layout_density="compact",
                formatting_instructions=(
                    "FORMAT AS A COMPACT, CLEAN RESPONSE:\n"
                    "- Provide a direct, concise 1 to 3 sentence answer.\n"
                    "- Do NOT use unnecessary headings, cards, bulleted lists, or verbose padding.\n"
                    "- Keep it conversational, formal, and crystal clear."
                ),
            )

        # 2. Institutional Transition (VVIT -> VVITU)
        if cls._RE_TRANSITION.search(q):
            return ResponsePlan(
                response_type="current_historical",
                title_hint="Institutional Evolution (VVIT → VVITU)",
                layout_density="expanded",
                formatting_instructions=(
                    "FORMAT AS AN INSTITUTIONAL TRANSITION SUMMARY:\n"
                    "- Start with an evolution timeline:\n"
                    "  1. **Legacy Era (VVIT)**: Founded as Vasireddy Venkatadri Institute of Technology (affiliated institute).\n"
                    "  2. **Transition**: Upgraded and gained university autonomy.\n"
                    "  3. **Current Era (VVITU)**: Operating as VVIT University (official portal: https://vvitu.ac.in/).\n"
                    "- Explicitly clarify the status of both portals: current is https://vvitu.ac.in/, while https://vvitguntur.com/ is preserved for archival records.\n"
                    "- Use clear, bold stage headings."
                ),
            )

        # 3. Direct Factual Lookups (Principal, Website, Address, etc.)
        if cls._RE_FACTUAL.search(q):
            return ResponsePlan(
                response_type="factual",
                title_hint="Authoritative Fact",
                layout_density="compact",
                formatting_instructions=(
                    "FORMAT AS A PROMINENT FACTUAL LOOKUP:\n"
                    "- First line: State the single primary fact clearly and prominently in bold (e.g. **Official Website: https://vvitu.ac.in/** or **Principal: Dr. ...**).\n"
                    "- Second block: Add 1 or 2 brief supporting sentences providing context or official role.\n"
                    "- Clearly mention the institutional era (Current VVITU vs Legacy VVIT).\n"
                    "- Avoid long explanations or filler paragraphs."
                ),
            )

        # 4. Code / Programming
        if cls._RE_CODE.search(q):
            return ResponsePlan(
                response_type="code",
                title_hint="Code Implementation",
                layout_density="expanded",
                formatting_instructions=(
                    "FORMAT AS A DEVELOPER CODE SOLUTION:\n"
                    "- Brief 1-sentence introduction of the approach.\n"
                    "- Complete, clean, well-commented code block using standard markdown ```language fences.\n"
                    "- Bulleted explanation of key lines, time/space complexity, and sample input/output."
                ),
            )

        # 5. Calculation / Numerical Math
        if cls._RE_CALCULATION.search(q):
            return ResponsePlan(
                response_type="calculation",
                title_hint="Step-by-Step Calculation",
                layout_density="medium",
                formatting_instructions=(
                    "FORMAT AS A STEP-BY-STEP CALCULATION:\n"
                    "1. **Formula**: State the underlying equation clearly.\n"
                    "2. **Given Values**: List input variables and values.\n"
                    "3. **Substitution & Steps**: Show mathematical substitution step-by-step.\n"
                    "4. **Final Result**: State the calculated result prominently in bold."
                ),
            )

        # 6. Comparison (e.g. CSE vs ECE)
        if cls._RE_COMPARISON.search(q):
            return ResponsePlan(
                response_type="comparison",
                title_hint="Side-by-Side Comparison",
                layout_density="expanded",
                formatting_instructions=(
                    "FORMAT AS A STRUCTURED COMPARISON TABLE:\n"
                    "- Start with a 1-sentence context statement.\n"
                    "- Create a markdown table comparing attributes across entities:\n"
                    "  | Feature / Aspect | Entity A | Entity B |\n"
                    "  | :--- | :--- | :--- |\n"
                    "  Include rows for Core Focus, Key Subjects, Labs/Tools, Career Scope, etc.\n"
                    "- Follow with a brief 2-bullet summary highlighting key selection considerations."
                ),
            )

        # 7. Checklists (Documents Required, Eligibility)
        if cls._RE_CHECKLIST.search(q):
            return ResponsePlan(
                response_type="checklist",
                title_hint="Required Checklist",
                layout_density="medium",
                formatting_instructions=(
                    "FORMAT AS A STRUCTURED CHECKLIST:\n"
                    "- Start with an introduction of who needs these requirements.\n"
                    "- Use markdown checklist items:\n"
                    "  - [ ] Item 1 (with brief requirement details)\n"
                    "  - [ ] Item 2 (with brief requirement details)\n"
                    "- Group into logical categories (e.g., Academic Certificates, Identity Proofs, Quota Documents) if more than 5 items.\n"
                    "- Include a short reminder note about verification and original vs copies."
                ),
            )

        # 8. Schedules & Calendars
        if cls._RE_SCHEDULE.search(q):
            return ResponsePlan(
                response_type="schedule",
                title_hint="Schedule & Important Dates",
                layout_density="medium",
                formatting_instructions=(
                    "FORMAT AS A STRUCTURED TIMETABLE / SCHEDULE:\n"
                    "- Present events in chronological order using a markdown table:\n"
                    "  | Event / Milestone | Scheduled Date / Window | Details |\n"
                    "  | :--- | :--- | :--- |\n"
                    "- If exact dates are not yet published, explicitly indicate verification status and cite where students can check for updates (e.g. https://vvitu.ac.in/).\n"
                    "- Highlight upcoming deadlines."
                ),
            )

        # 9. Fee Structures
        if cls._RE_FEE.search(q):
            return ResponsePlan(
                response_type="fee",
                title_hint="Fee Structure & Payment Details",
                layout_density="medium",
                formatting_instructions=(
                    "FORMAT AS A STRUCTURED FEE TABLE:\n"
                    "- Present fees in a clear markdown table:\n"
                    "  | Fee Category | Amount | Period / Quota | Notes |\n"
                    "  | :--- | :--- | :--- | :--- |\n"
                    "- Strictly separate CURRENT VVITU university fees from HISTORICAL VVIT institute records.\n"
                    "- Mention payment methods or scholarship/concession policies if available in context."
                ),
            )

        # 10. Process / Workflow (Admissions, Procedures)
        if cls._RE_PROCESS.search(q):
            return ResponsePlan(
                response_type="process",
                title_hint="Step-by-Step Procedure",
                layout_density="expanded",
                formatting_instructions=(
                    "FORMAT AS A SEQUENTIAL NUMBERED PROCESS:\n"
                    "- Number each step sequentially:\n"
                    "  1. **Step Name**: Clear description of what the student needs to do.\n"
                    "  2. **Step Name**: Actions, portal links, or offices to contact.\n"
                    "- Include prerequisites at the beginning and the final confirmation step at the end.\n"
                    "- Keep steps actionable and student-friendly."
                ),
            )

        # 11. Academic Exam Answer (10-mark style)
        if cls._RE_ACADEMIC_EXAM.search(q):
            return ResponsePlan(
                response_type="academic_exam",
                title_hint="Academic Exam Response",
                layout_density="expanded",
                formatting_instructions=(
                    "FORMAT AS A HIGH-SCORING ACADEMIC EXAM ANSWER:\n"
                    "## 1. Formal Definition & Overview\n"
                    "Concise, textbook-accurate definition.\n\n"
                    "## 2. Working Principle & Architecture\n"
                    "Core mechanics and theory.\n\n"
                    "## 3. Step-by-Step Algorithm / Steps\n"
                    "Numbered algorithmic sequence.\n\n"
                    "## 4. Concrete Example / Application\n"
                    "Real-world or illustrative example.\n\n"
                    "## 5. Key Advantages & Limitations\n"
                    "- Pros and cons in bullet points.\n\n"
                    "## 6. Exam-Ready Conclusion\n"
                    "Final 1-sentence recap for exams."
                ),
            )

        # 12. Structured Lists (Branches, Courses, Faculty)
        if cls._RE_LIST.search(q):
            return ResponsePlan(
                response_type="list",
                title_hint="Program & Course Catalog",
                layout_density="medium",
                formatting_instructions=(
                    "FORMAT AS A CLEAN STRUCTURED LIST:\n"
                    "- Categorize items by department or degree level (e.g. Undergraduate, Postgraduate).\n"
                    "- Present items with bullet points and brief 1-line descriptions or specializations.\n"
                    "- Do not produce walls of unbroken text; use clean indentation and bold item names."
                ),
            )

        # 13. Educational / Concept Explanation (Teaching Mode)
        if cls._RE_EDUCATIONAL.search(q):
            return ResponsePlan(
                response_type="educational",
                title_hint="Concept Explanation & Teaching Guide",
                layout_density="expanded",
                formatting_instructions=(
                    "FORMAT AS A 5-STAGE TEACHING EXPLANATION:\n"
                    "Adopt an engaging, academic teaching style:\n"
                    "### 1. Simple Definition & Intuition\n"
                    "Explain the core idea simply so a beginner immediately grasps it.\n\n"
                    "### 2. Why It Matters\n"
                    "Why was this developed and what problem does it solve?\n\n"
                    "### 3. How It Works (Core Mechanics)\n"
                    "Step-by-step breakdown of the mechanics or architecture.\n\n"
                    "### 4. Real-World Example\n"
                    "A relatable example demonstrating practical use.\n\n"
                    "### 5. Key Takeaways\n"
                    "- 3 to 4 essential points to remember."
                ),
            )

        # 14. Historical Query
        if cls._RE_HISTORICAL.search(q):
            return ResponsePlan(
                response_type="historical",
                title_hint="Historical Institute Records (VVIT)",
                layout_density="medium",
                formatting_instructions=(
                    "FORMAT AS A HISTORICAL RECORD:\n"
                    "- Clearly label the information as representing the legacy VVIT institute era.\n"
                    "- Provide chronological context and archival milestones.\n"
                    "- Note the later transition to VVIT University (VVITU)."
                ),
            )

        # 15. Default Informational
        return ResponsePlan(
            response_type="informational",
            title_hint="Institutional Information",
            layout_density="medium",
            formatting_instructions=(
                "FORMAT AS A CLEAN, STRUCTURED ANSWER:\n"
                "- Lead with a clear, direct answer to the question.\n"
                "- Organize supporting details using clean headings or bullet points.\n"
                "- Clearly indicate source era (VVITU current vs VVIT legacy) when mentioning facts."
            ),
        )


response_planner = ResponsePlanner()
