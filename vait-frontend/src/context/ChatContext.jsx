import { createContext, useContext, useState, useCallback } from 'react';
import { streamMessageToVAIT, detectCategory } from '../utils/mockAI';

const ChatContext = createContext(null);

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function groupConversations(conversations) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekStart = new Date(today);
  weekStart.setDate(weekStart.getDate() - 7);
  const monthStart = new Date(today);
  monthStart.setDate(monthStart.getDate() - 30);

  const groups = {
    Today: [],
    Yesterday: [],
    'This Week': [],
    'This Month': [],
    Older: [],
  };

  conversations.forEach((conv) => {
    const d = new Date(conv.createdAt);
    if (d >= today) groups.Today.push(conv);
    else if (d >= yesterday) groups.Yesterday.push(conv);
    else if (d >= weekStart) groups['This Week'].push(conv);
    else if (d >= monthStart) groups['This Month'].push(conv);
    else groups.Older.push(conv);
  });

  return groups;
}

export function ChatProvider({ children }) {
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [department, setDepartment] = useState('CSE');
  const [academicYear, setAcademicYear] = useState('2025-26');
  const [categoryFilter, setCategoryFilter] = useState('All');

  const activeConversation = conversations.find((c) => c.id === activeConversationId) || null;

  const createConversation = useCallback(() => {
    const id = generateId();
    const conv = {
      id,
      title: 'New Conversation',
      messages: [],
      category: 'Academic',
      confidence: null,
      sources: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setConversations((prev) => [conv, ...prev]);
    setActiveConversationId(id);
    return id;
  }, []);

  const sendMessage = useCallback(
    async (text) => {
      let convId = activeConversationId;

      if (!convId) {
        convId = generateId();
        const conv = {
          id: convId,
          title: text.slice(0, 50) + (text.length > 50 ? '...' : ''),
          messages: [],
          category: detectCategory(text),
          confidence: null,
          sources: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        setConversations((prev) => [conv, ...prev]);
        setActiveConversationId(convId);
      }

      const userMessage = {
        id: generateId(),
        role: 'user',
        text,
        timestamp: new Date().toISOString(),
      };

      const aiMessageId = generateId();

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const isFirst = c.messages.length === 0;
          return {
            ...c,
            title: isFirst ? text.slice(0, 50) + (text.length > 50 ? '...' : '') : c.title,
            category: isFirst ? detectCategory(text) : c.category,
            messages: [...c.messages, userMessage, { id: aiMessageId, role: 'assistant', text: '', isGenerating: true }],
            updatedAt: new Date().toISOString(),
          };
        })
      );

      const existingConv = conversations.find((c) => c.id === convId);
      const history = existingConv ? [...existingConv.messages] : [];

      setLoading(true);

      try {
        await streamMessageToVAIT({
          message: text,
          department,
          academicYear,
          history,
          onUpdate: (state) => {
            setConversations((prev) =>
              prev.map((c) => {
                if (c.id !== convId) return c;
                return {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === aiMessageId
                      ? {
                          ...m,
                          text: state.text,
                          heading: state.heading,
                          bullets: state.bullets,
                          sources: state.sources,
                          confidence: state.confidence,
                          category: state.category,
                          isGenerating: state.isGenerating,
                        }
                      : m
                  ),
                  confidence: state.confidence,
                  sources: state.sources,
                  updatedAt: new Date().toISOString(),
                };
              })
            );
          },
        });
      } finally {
        setLoading(false);
      }
    },
    [activeConversationId, department, academicYear]
  );

  const deleteConversation = useCallback(
    (id) => {
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        setActiveConversationId(null);
      }
    },
    [activeConversationId]
  );

  const renameConversation = useCallback((id, newTitle) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c))
    );
  }, []);

  const clearChat = useCallback(() => {
    if (!activeConversationId) return;
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeConversationId ? { ...c, messages: [], updatedAt: new Date().toISOString() } : c
      )
    );
  }, [activeConversationId]);

  const exportConversation = useCallback(() => {
    if (!activeConversation) return;
    const data = JSON.stringify(activeConversation, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vait-conversation-${activeConversation.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [activeConversation]);

  const groupedConversations = groupConversations(
    categoryFilter === 'All'
      ? conversations
      : conversations.filter((c) => c.category === categoryFilter)
  );

  const value = {
    conversations,
    groupedConversations,
    activeConversation,
    activeConversationId,
    loading,
    department,
    academicYear,
    categoryFilter,
    setActiveConversationId,
    setDepartment,
    setAcademicYear,
    setCategoryFilter,
    createConversation,
    sendMessage,
    deleteConversation,
    renameConversation,
    clearChat,
    exportConversation,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used within ChatProvider');
  return ctx;
}
