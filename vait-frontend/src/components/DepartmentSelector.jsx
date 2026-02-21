import { useChat } from '../context/ChatContext';

const DEPARTMENTS = ['CSE', 'AI', 'ECE', 'IT', 'Mechanical', 'Civil'];

export default function DepartmentSelector() {
  const { department, setDepartment } = useChat();

  return (
    <select
      className="topbar-select"
      value={department}
      onChange={(e) => setDepartment(e.target.value)}
      aria-label="Select department"
    >
      {DEPARTMENTS.map((d) => (
        <option key={d} value={d}>{d}</option>
      ))}
    </select>
  );
}
