import { useChat } from '../context/ChatContext';
import './TopBar.css';

const DEPARTMENTS = ['CSE', 'AI', 'ECE', 'IT', 'Mechanical', 'Civil'];
const YEARS = ['2023-24', '2024-25', '2025-26'];

export default function TopBar() {
  const {
    department,
    setDepartment,
    academicYear,
    setAcademicYear,
    clearChat,
    exportConversation,
    activeConversation,
  } = useChat();

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1 className="topbar-title">VAIT</h1>
        <span className="topbar-subtitle">VVIT's Official  Intelligence Assistant</span>
      </div>

      <div className="topbar-center">
        <span className="topbar-badge">Official Academic Mode</span>
      </div>

      <div className="topbar-right">
        <div className="topbar-select-group">
          <label className="topbar-label" htmlFor="dept-select">Dept</label>
          <select
            id="dept-select"
            className="topbar-select"
            value={department}
            onChange={(e) => setDepartment(e.target.value)}
          >
            {DEPARTMENTS.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>

        <div className="topbar-select-group">
          <label className="topbar-label" htmlFor="year-select">Year</label>
          <select
            id="year-select"
            className="topbar-select"
            value={academicYear}
            onChange={(e) => setAcademicYear(e.target.value)}
          >
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        {activeConversation && (
          <>
            <button className="topbar-btn" onClick={exportConversation} title="Export conversation">
              Export
            </button>
            <button className="topbar-btn topbar-btn-clear" onClick={clearChat} title="Clear chat">
              Clear
            </button>
          </>
        )}
      </div>
    </header>
  );
}
