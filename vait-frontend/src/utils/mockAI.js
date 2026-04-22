import { BASE_URL, buildApiUrl } from './apiConfig';

const CATEGORIES = ['Academic', 'Exams', 'Administration', 'Placements'];

const MOCK_SOURCES = {
  Academic: [
    'academic-calendar-2025-26.pdf',
    'regulations-r23.pdf',
  ],
  Exams: [
    'exam-schedule-sem2-2026.pdf',
    'examination-regulations.pdf',
  ],
  Administration: [
    'fee-structure-2025-26.pdf',
    'hostel-guidelines.pdf',
  ],
  Placements: [
    'placement-report-2025.pdf',
    'company-visits-schedule.pdf',
  ],
};

const MOCK_RESPONSES = {
  Academic: [
    {
      heading: 'Academic Calendar — 2025-26',
      body: 'The academic year 2025-26 commenced on July 15, 2025. The first semester concludes on December 20, 2025, followed by semester examinations from January 5 to January 20, 2026.',
      bullets: [
        'First semester: July 15 – December 20, 2025',
        'Mid-semester exams: September 22 – September 27, 2025',
        'Second semester: January 25 – June 15, 2026',
        'Summer break: June 16 – July 14, 2026',
      ],
    },
    {
      heading: 'Course Regulations — R23',
      body: 'Under the R23 regulation, students must complete a minimum of 160 credits across eight semesters. Each semester comprises a combination of theory courses, laboratory sessions, and project work.',
      bullets: [
        'Minimum attendance requirement: 75%',
        'Maximum credits per semester: 24',
        'Mandatory internship in 6th semester',
        'Capstone project spans 7th and 8th semesters',
      ],
    },
  ],
  Exams: [
    {
      heading: 'Examination Schedule — Semester II, 2026',
      body: 'The second semester examinations for the academic year 2025-26 are scheduled to begin on June 1, 2026. Students are advised to collect their hall tickets from the examination branch by May 25, 2026.',
      bullets: [
        'Exam commencement: June 1, 2026',
        'Hall ticket distribution: May 25, 2026',
        'Supplementary exams: July 15, 2026',
        'Results announcement: Within 30 days of last exam',
      ],
    },
  ],
  Administration: [
    {
      heading: 'Fee Structure — 2025-26',
      body: 'The tuition fee for the academic year 2025-26 has been set in accordance with the fee regulatory committee guidelines. Payment can be made online through the student portal or at the accounts section.',
      bullets: [
        'Tuition fee: As per category (A/B/C)',
        'Payment deadline: Within 15 days of admission',
        'Scholarship applications: Open until August 30',
        'Fee concession: Available for merit students',
      ],
    },
  ],
  Placements: [
    {
      heading: 'Placement Statistics — 2025',
      body: 'The Training and Placement Cell successfully facilitated campus recruitment drives for the 2024-25 graduating batch. A total of 87% of eligible students received placement offers.',
      bullets: [
        'Total companies visited: 120+',
        'Highest package: 42 LPA',
        'Average package: 6.8 LPA',
        'Top recruiters: TCS, Infosys, Wipro, Amazon, Microsoft',
      ],
    },
  ],
};

function detectCategory(message) {
  const lower = message.toLowerCase();
  if (/exam|test|hall ticket|suppl|result|grade|marks|cgpa/i.test(lower)) return 'Exams';
  if (/place|recruit|company|package|offer|intern|job/i.test(lower)) return 'Placements';
  if (/fee|hostel|bus|admin|library|id card|certificate/i.test(lower)) return 'Administration';
  return 'Academic';
}

function pickConfidence() {
  const roll = Math.random();
  if (roll > 0.6) return 'High';
  if (roll > 0.25) return 'Medium';
  return 'Low';
}

/**
 * Send user message to the VAIT backend API.
 * Falls back to mock response only if the backend is unreachable.
 */
export async function sendMessageToVAIT({ message, department, academicYear }) {
  console.log('[VAIT] Sending message:', message);
  const endpoint = buildApiUrl('/api/vait/chat');
  console.log('[VAIT] API base URL:', BASE_URL);
  console.log('[VAIT] API endpoint:', endpoint);

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        department: department || null,
        academic_year: academicYear || null,
      }),
    });

    if (!res.ok) {
      throw new Error(`Backend returned ${res.status}`);
    }

    const data = await res.json();
    console.log('[VAIT] Backend response:', data);

    const category = data.intent || detectCategory(message);

    return {
      text: data.reply,
      heading: null,
      bullets: null,
      sources: data.sources || [],
      confidence: data.confidence || 'Low',
      category,
      department: department || 'General',
      academicYear: academicYear || '2025-26',
      timestamp: new Date().toISOString(),
    };
  } catch (err) {
    console.error('[VAIT] Backend call failed, using mock fallback:', err);

    // ── Fallback to mock (development only) ─────────────────────
    const category = detectCategory(message);
    const responses = MOCK_RESPONSES[category] || MOCK_RESPONSES['Academic'];
    const selected = responses[Math.floor(Math.random() * responses.length)];
    const sources = MOCK_SOURCES[category] || MOCK_SOURCES['Academic'];

    return {
      text: selected.body,
      heading: selected.heading,
      bullets: selected.bullets,
      sources,
      confidence: pickConfidence(),
      category,
      department: department || 'General',
      academicYear: academicYear || '2025-26',
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Stream user message from the VAIT backend API.
 */
export async function streamMessageToVAIT({ message, department, academicYear, onUpdate }) {
  console.log('[VAIT] Streaming message:', message);
  const endpoint = buildApiUrl('/api/vait/chat/stream');

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        department: department || null,
        academic_year: academicYear || null,
      }),
    });

    if (!res.ok) {
      throw new Error(`Backend returned ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    let state = {
      text: '',
      heading: null,
      bullets: null,
      sources: [],
      confidence: 'Low',
      category: detectCategory(message),
      department: department || 'General',
      academicYear: academicYear || '2025-26',
      timestamp: new Date().toISOString(),
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim().startsWith('data: ')) {
          const dataStr = line.replace(/^data:\s*/, '').trim();
          if (!dataStr || dataStr === '[DONE]') continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (parsed.type === 'metadata') {
              state.sources = parsed.sources || [];
              state.confidence = parsed.confidence || 'Low';
              if (parsed.intent) state.category = parsed.intent;
              onUpdate({ ...state, isGenerating: true });
            } else if (parsed.type === 'content') {
              state.text += parsed.content;
              onUpdate({ ...state, isGenerating: true });
            } else if (parsed.type === 'fallback_triggered') {
              state.text = ''; // Clear output on fallback to avoid duplicates
              onUpdate({ ...state, isGenerating: true });
            } else if (parsed.type === 'error') {
              state.text += `\n\n[Error: ${parsed.error}]`;
              onUpdate({ ...state, isGenerating: false });
            } else if (parsed.type === 'done') {
              onUpdate({ ...state, isGenerating: false });
            }
          } catch (e) {
            console.error('Error parsing stream chunk', e, dataStr);
          }
        }
      }
    }

    if (buffer.trim().startsWith('data: ')) {
      try {
        const parsed = JSON.parse(buffer.replace(/^data:\s*/, '').trim());
        if (parsed.type === 'content') state.text += parsed.content;
      } catch (e) {}
    }
    onUpdate({ ...state, isGenerating: false });
    return state;
  } catch (err) {
    console.error('[VAIT] Stream failed, using fallback:', err);
    const category = detectCategory(message);
    const responses = MOCK_RESPONSES[category] || MOCK_RESPONSES['Academic'];
    const selected = responses[Math.floor(Math.random() * responses.length)];
    
    let state = {
      text: '',
      heading: selected.heading,
      bullets: selected.bullets,
      sources: MOCK_SOURCES[category] || MOCK_SOURCES['Academic'],
      confidence: pickConfidence(),
      category,
      department: department || 'General',
      academicYear: academicYear || '2025-26',
      timestamp: new Date().toISOString(),
      isGenerating: true
    };

    const mockText = selected.body;
    for (let i = 0; i < mockText.length; i += 5) {
      state.text += mockText.substr(i, 5);
      onUpdate({ ...state, isGenerating: true });
      await new Promise(r => setTimeout(r, 20));
    }
    onUpdate({ ...state, isGenerating: false });
    return state;
  }
}

export { detectCategory, CATEGORIES };
