import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { sendMessageToVAIT, detectCategory } from '../utils/mockAI';
import { buildApiUrl } from '../utils/apiConfig';
import { useAuth } from './AuthContext';

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

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function normalizeConversation(conv) {
  return {
    id: conv.id,
    title: conv.title || 'New Conversation',
    messages: (conv.messages || []).map((message) => ({
      ...message,
      text: message.text || '',
      timestamp: message.timestamp || new Date().toISOString(),
    })),
    category: conv.category || 'Academic',
    confidence: conv.confidence || null,
    sources: conv.sources || [],
    createdAt: conv.createdAt || new Date().toISOString(),
    updatedAt: conv.updatedAt || new Date().toISOString(),
  };
}

export function ChatProvider({ children }) {
  const { token, isAuthenticated } = useAuth();
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [department, setDepartment] = useState('CSE');
  const [academicYear, setAcademicYear] = useState('2025-26');
  const [categoryFilter, setCategoryFilter] = useState('All');

  const activeConversation = conversations.find((c) => c.id === activeConversationId) || null;

  useEffect(() => {
    let alive = true;

    async function loadConversations() {
      if (!isAuthenticated || !token) {
        setConversations([]);
        setActiveConversationId(null);
        return;
      }

      try {
        const response = await fetch(buildApiUrl('/api/vait/chat/conversations'), {
          headers: authHeaders(token),
        });
        const data = await readJson(response);
        if (!response.ok) {
          throw new Error(data?.detail || 'Could not load conversations.');
        }
        if (!alive) return;
        const nextConversations = (data?.conversations || []).map(normalizeConversation);
        setConversations(nextConversations);
        setActiveConversationId((current) =>
          current && nextConversations.some((conv) => conv.id === current)
            ? current
            : nextConversations[0]?.id || null
        );
      } catch (error) {
        console.error('[VAIT][Debug] Failed to load conversations:', error);
      }
    }

    loadConversations();
    return () => {
      alive = false;
    };
  }, [isAuthenticated, token]);

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
      const assistantTimestamp = new Date().toISOString();

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const isFirst = c.messages.length === 0;
          return {
            ...c,
            title: isFirst ? text.slice(0, 50) + (text.length > 50 ? '...' : '') : c.title,
            category: isFirst ? detectCategory(text) : c.category,
            messages: [
              ...c.messages,
              userMessage,
              {
                id: aiMessageId,
                role: 'assistant',
                text: '',
                isGenerating: true,
                timestamp: assistantTimestamp,
              },
            ],
            updatedAt: new Date().toISOString(),
          };
        })
      );

      setLoading(true);

      try {
        const currentConversation = conversations.find((c) => c.id === convId);
        const state = await sendMessageToVAIT({
          message: text,
          department,
          academicYear,
          history: currentConversation?.messages || [],
          conversationId: convId,
          token,
        });

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
                      isGenerating: false,
                      timestamp: state.timestamp || m.timestamp,
                    }
                  : m
              ),
              confidence: state.confidence,
              sources: state.sources,
              updatedAt: new Date().toISOString(),
            };
          })
        );
      } catch (error) {
        const errorText = error?.message || 'Unable to process your request right now.';
        console.error('[VAIT][Debug] Chat request failed:', errorText);

        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convId) return c;
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === aiMessageId
                  ? {
                      ...m,
                      text: `Error: ${errorText}`,
                      isGenerating: false,
                      confidence: 'Low',
                      sources: [],
                      timestamp: m.timestamp || assistantTimestamp,
                    }
                  : m
              ),
              confidence: 'Low',
              sources: [],
              updatedAt: new Date().toISOString(),
            };
          })
        );
      } finally {
        setLoading(false);
      }
    },
    [academicYear, activeConversationId, conversations, department, token]
  );

  const deleteConversation = useCallback(
    (id) => {
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        setActiveConversationId(null);
      }
      if (token) {
        fetch(buildApiUrl(`/api/vait/chat/conversations/${encodeURIComponent(id)}`), {
          method: 'DELETE',
          headers: authHeaders(token),
        }).catch((error) => console.error('[VAIT][Debug] Failed to delete conversation:', error));
      }
    },
    [activeConversationId, token]
  );

  const renameConversation = useCallback((id, newTitle) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c))
    );
    if (token) {
      fetch(buildApiUrl(`/api/vait/chat/conversations/${encodeURIComponent(id)}`), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', ...authHeaders(token) },
        body: JSON.stringify({ title: newTitle }),
      }).catch((error) => console.error('[VAIT][Debug] Failed to rename conversation:', error));
    }
  }, [token]);

  const clearChat = useCallback(() => {
    if (!activeConversationId) return;
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeConversationId ? { ...c, messages: [], updatedAt: new Date().toISOString() } : c
      )
    );
    if (token) {
      fetch(buildApiUrl(`/api/vait/chat/conversations/${encodeURIComponent(activeConversationId)}/clear`), {
        method: 'POST',
        headers: authHeaders(token),
      }).catch((error) => console.error('[VAIT][Debug] Failed to clear conversation:', error));
    }
  }, [activeConversationId, token]);

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
