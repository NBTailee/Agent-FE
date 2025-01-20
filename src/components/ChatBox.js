import styled from "styled-components";
const Container = styled.div`
  width: 50%;
  height: 60%;
  padding: 10px;
  word-wrap: break-word;
  font-size: 16px;
  padding: 10px;
  overflow: auto;
  box-sizing: border-box;
  scrollbar-width: none;
`;

function ChatBox({ children }) {
  return <Container>{children}</Container>;
}

export default ChatBox;
