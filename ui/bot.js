import React, { useState, useEffect, useRef } from "react";
import "./Bot.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Welcome to TechRacine! How can I assist you today?" },
  ]);
  const [userInput, setUserInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const chatBodyRef = useRef(null);

  useEffect(() => {
    chatBodyRef.current?.scrollTo(0, chatBodyRef.current.scrollHeight);
  }, [messages]);

  const sendMessage = async (query) => {
    if (!query.trim()) return;

    const newMessage = { sender: "user", text: query };
    setMessages((prev) => [...prev, newMessage]);
    setUserInput("");
    setIsTyping(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!response.body) throw new Error("No response body found");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let botMessage = "";
      let updatedBotMessage = { sender: "bot", text: "" };

      setMessages((prev) => [...prev, updatedBotMessage]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        botMessage += decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const updatedMessages = [...prev];
          updatedMessages[updatedMessages.length - 1] = { sender: "bot", text: botMessage };
          return updatedMessages;
        });
      }
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setMessages((prev) => [...prev, { sender: "bot", text: "Error fetching response." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSendClick = () => sendMessage(userInput);

  const handleKeyPress = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage(userInput);
    }
  };

  const handleQuickReply = (question) => sendMessage(question);

  return (
    <div className="chatbot-container">
      <div className="chat-header">AskRacine</div>
      <div className="chat-body" ref={chatBodyRef}>
        {messages.map((message, index) => (
          <div key={index} className={`chat-message ${message.sender}-message`}>
            {message.sender === "bot" && <span className="material-symbols-outlined bot-icon">smart_toy</span>}
            <span dangerouslySetInnerHTML={{ __html: message.text }}></span>

          </div>
        ))}
        {isTyping && <div className="typing-indicator">AskRacine is typing...</div>}
      </div>
      <div className="chat-footer">
        <textarea
          placeholder="Type your message..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={handleKeyPress}
        />
        <button onClick={handleSendClick}>Send</button>
      </div>

      {/* Quick reply buttons */}
      <div className="quick-replies">
        {["What kind of products do you develop?", "How can I contact sales?", "What services do you offer?"].map(
          (question, index) => (
            <button key={index} className="custom-question" onClick={() => handleQuickReply(question)}>
              {question}
            </button>
          )
        )}
      </div>
    </div>
  );
};

export default Chatbot;
