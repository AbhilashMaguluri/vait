import { useChat } from '../context/ChatContext';

const YEARS = ['2023-24', '2024-25', '2025-26'];

export default function YearSelector() {
  const { academicYear, setAcademicYear } = useChat();

  return (
    <select
      className="topbar-select"
      value={academicYear}
      onChange={(e) => setAcademicYear(e.target.value)}
      aria-label="Select academic year"
    >
      {YEARS.map((y) => (
        <option key={y} value={y}>{y}</option>
      ))}
    </select>
  );
}
