import "./App.css";
import ResponseBox from "./components/ResponseBox";
import UserBox from "./components/UserBox";
import ChatBox from "./components/ChatBox";
import NavBar from "./components/NavBar";
import { useState, useRef, useEffect } from "react";
import { BeatLoader } from "react-spinners";

function App() {
  const [history, setHistory] = useState([
    { user: "Im Tai Big Dick", agent: "Oke I know that bro" },
  ]);
  const chatEndRef = useRef(null);
  const [message, setMessage] = useState("");

  const scrollToBottom = () => {
    chatEndRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [history]);
  return (
    <div className="App">
      <NavBar />
      <h1 className="header">TAIGPT</h1>
      <ChatBox>
        {history.map((res, index) => (
          <div key={index}>
            <UserBox text={res.user} />
            {res.agent ? (
              <ResponseBox text={res.agent} />
            ) : (
              <BeatLoader size={13} color="#ececec" />
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </ChatBox>
      <form>
        <textarea
          placeholder="Enter your message here...."
          value={message}
          className="message-input"
          type="text"
          required
          onChange={(e) => {
            e.preventDefault();
            setMessage(e.target.value);
          }}
          // onKeyDown={handleKeyDown}
        />
        <div className="btn-group">
          <button
            className="submit-btn"
            // onClick={(e) => {
            //   e.preventDefault();
            //   handleMessage(message, setMessage, setHistory);
            // }}
          >
            SUBMIT
          </button>
          <button
            className="clear-btn"
            onClick={() => {
              setHistory([]);
            }}
          >
            CLEAR HISTORY
          </button>
        </div>
      </form>
    </div>
  );
}

export default App;
