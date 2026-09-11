import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { sendMessageToVAIT, streamMessageToVAIT, detectCategory } from '../utils/mockAI';
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
      responseType: message.response_type || message.responseType || 'informational',
      structuredSources: message.structured_sources || message.structuredSources || [],
      isGenerating: false,
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
    async (text, retryAssistantId = null) => {
      let convId = activeConversationId;
      const aiMessageId = retryAssistantId || generateId();
      const assistantTimestamp = new Date().toISOString();

      if (!retryAssistantId) {
        const userMessage = {
          id: generateId(),
          role: 'user',
          text,
          timestamp: new Date().toISOString(),
        };

        const initialAssistantMessage = {
          id: aiMessageId,
          role: 'assistant',
          text: '',
          isGenerating: true,
          status: 'thinking',
          statusText: 'VAIT is thinking...',
          responseType: 'informational',
          structuredSources: [],
          sourceVisibility: 'none',
          sources: [],
          confidence: null,
          originalQuery: text,
          timestamp: assistantTimestamp,
        };

        if (!convId) {
          convId = generateId();
          const conv = {
            id: convId,
            title: text.slice(0, 50) + (text.length > 50 ? '...' : ''),
            messages: [userMessage, initialAssistantMessage],
            category: detectCategory(text),
            confidence: null,
            sources: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          };
          setConversations((prev) => [conv, ...prev]);
          setActiveConversationId(convId);
        } else {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              return {
                ...c,
                messages: [...c.messages, userMessage, initialAssistantMessage],
                updatedAt: new Date().toISOString(),
              };
            })
          );
        }
      } else {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convId) return c;
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === aiMessageId
                  ? {
                      ...m,
                      text: '',
                      error: null,
                      isGenerating: true,
                      status: 'thinking',
                      statusText: 'VAIT is thinking...',
                      timestamp: assistantTimestamp,
                    }
                  : m
              ),
              updatedAt: new Date().toISOString(),
            };
          })
        );
      }

      setLoading(true);

      try {
        const currentConversation = conversations.find((c) => c.id === convId);
        const historyForLLM = (currentConversation?.messages || []).filter(
          (m) => m.id !== aiMessageId
        );

        const finalState = await streamMessageToVAIT({
          message: text,
          department,
          academicYear,
          history: historyForLLM,
          conversationId: convId,
          token,
          onUpdate: (streamState) => {
            setConversations((prev) =>
              prev.map((c) => {
                if (c.id !== convId) return c;
                return {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === aiMessageId
                      ? {
                          ...m,
                          text: streamState.text,
                          heading: streamState.heading || m.heading,
                          bullets: streamState.bullets || m.bullets,
                          sources: streamState.sources || m.sources,
                          confidence: streamState.confidence || m.confidence,
                          category: streamState.category || m.category,
                          responseType: streamState.responseType || m.responseType,
                          structuredSources: streamState.structuredSources || m.structuredSources,
                          sourceVisibility: streamState.sourceVisibility || m.sourceVisibility || 'none',
                          isGenerating: streamState.isGenerating,
                          status: streamState.text?.trim() ? 'streaming' : 'thinking',
                          statusText: streamState.text?.trim() ? null : 'VAIT is thinking...',
                          error: null,
                          timestamp: streamState.timestamp || m.timestamp,
                        }
                      : m
                  ),
                  confidence: streamState.confidence || c.confidence,
                  sources: streamState.sources?.length ? streamState.sources : c.sources,
                  updatedAt: new Date().toISOString(),
                };
              })
            );
          },
        });

        const replyText = finalState?.text?.trim()
          ? finalState.text
          : "I couldn't generate a response. Please try again.";

        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convId) return c;
            return {
              ...c,
              messages: c.messages.map((m) =>
                m.id === aiMessageId
                  ? {
                      ...m,
                      text: replyText,
                      heading: finalState?.heading || null,
                      bullets: finalState?.bullets || null,
                      sources: finalState?.sources || [],
                      confidence: finalState?.confidence || 'Low',
                      category: finalState?.category || m.category,
                      responseType: finalState?.responseType || m.responseType,
                      structuredSources: finalState?.structuredSources || m.structuredSources,
                      sourceVisibility: finalState?.sourceVisibility || m.sourceVisibility || 'none',
                      isGenerating: false,
                      status: 'complete',
                      statusText: null,
                      error: null,
                      timestamp: finalState?.timestamp || m.timestamp,
                    }
                  : m
              ),
              confidence: finalState?.confidence || c.confidence,
              sources: finalState?.sources?.length ? finalState.sources : c.sources,
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
                      text: '',
                      error: errorText,
                      originalQuery: text,
                      isGenerating: false,
                      status: 'error',
                      statusText: null,
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

  const retryMessage = useCallback(
    (message) => {
      if (!message) return;
      const query =
        message.originalQuery ||
        conversations
          .find((c) => c.id === activeConversationId)
          ?.messages.find((m, i, arr) => arr[i + 1]?.id === message.id && m.role === 'user')
          ?.text;
      if (query) {
        sendMessage(query, message.id);
      }
    },
    [activeConversationId, conversations, sendMessage]
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
    retryMessage,
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
