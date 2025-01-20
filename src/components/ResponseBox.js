import React from "react";
import styled from "styled-components";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";

const Container = styled.div`
  max-width: 70%;
  min-width: 10%;
  width: auto;
  box-sizing: border-box;
  padding: 8px;
  border: none;
  border-radius: 8px;
  word-wrap: break-word;
  margin: 20px 0px;
  font-size: 16px;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  color: #ececec;
`;

const newDiv = styled.div`
  white-space: pre-wrap;
  max-width: 70%;
  min-width: 10%;
  width: auto;
  background-color: #1e1e1e;
`;

function ResponseBox({ text }) {
  return (
    <Container>
      <newDiv>
        <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{text}</ReactMarkdown>
      </newDiv>
    </Container>
  );
}

export default ResponseBox;
