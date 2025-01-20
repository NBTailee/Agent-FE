import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: fit-content;
  min-width: 10%;
  width: auto;
  box-sizing: border-box;
  padding: 10px;
  color: white;
  background-color: #2f2f2f;
  border: none;
  border-radius: 8px;
  word-wrap: break-word;
  margin: 20px 0px;
  font-size: 16px;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  align-self: flex-end;
  margin-left: auto;
  color: #ececec;
`;

function UserBox({ text }) {
  return (
    <Container>
      <div>{text}</div>
    </Container>
  );
}

export default UserBox;
